import math
import numpy as np
import torch
from utils import ActionConverter, encode_state, get_valid_moves_mask

class TreeNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.q_value = 0.0
        self.w_sum = 0.0
        self.prior_p = prior_p

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, action_probs):
        for action_idx, prob in enumerate(action_probs):
            if prob > 0:
                self.children[action_idx] = TreeNode(self, prob)

    def update(self, leaf_value):
        self.n_visits += 1
        self.w_sum += leaf_value
        self.q_value = self.w_sum / self.n_visits

    def get_best_action(self):
        return max(self.children.items(), key=lambda item: item[1].n_visits)[0]

class MCTS:
    def __init__(self, model, c_puct=1.5, n_simulations=100):
        self.model = model
        self.c_puct = c_puct
        self.n_sims = n_simulations

    def search(self, game_env):
        """
        [预测用] 普通搜索，直接返回最佳动作
        """
        root = self.search_and_return_root(game_env)
        return ActionConverter.int_to_action(root.get_best_action())

    def search_and_return_root(self, game_env):
        """
        [训练用] 执行搜索，并返回根节点
        这样训练脚本才能读取 root.children 里的访问次数，生成概率分布 pi
        """
        root = TreeNode(None, 1.0)

        for _ in range(self.n_sims):
            virtual_env = game_env.clone()
            node = root

            # 1. Selection (选择)
            while not node.is_leaf():
                action_idx, node = self._select_child(node)
                action = ActionConverter.int_to_action(action_idx)
                virtual_env.step(action)

            # 2. Expand & Evaluate (扩展与评估)
            value = 0.0
            if virtual_env.game_over:
                if virtual_env.winner == virtual_env.current_player:
                    value = 1.0
                elif virtual_env.winner == 0:
                    value = 0.0
                else:
                    value = -1.0
            else:
                value = self._expand(node, virtual_env)

            # 3. Backpropagate (回溯)
            self._backprop(node, value)

        return root

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action_idx, child in node.children.items():
            u = self.c_puct * child.prior_p * math.sqrt(node.n_visits) / (1 + child.n_visits)
            score = child.q_value + u
            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child
        return best_action, best_child

    def _expand(self, node, env):
        state_tensor = encode_state(env)
        mask = get_valid_moves_mask(env)
        
        self.model.eval()
        with torch.no_grad():
            log_probs, v = self.model(state_tensor)
            
        probs = torch.exp(log_probs).numpy().flatten()
        probs = probs * mask.numpy()
        
        sum_p = np.sum(probs)
        if sum_p > 0:
            probs /= sum_p
        else:
            probs = mask.numpy() / np.sum(mask.numpy())
            
        node.expand(probs)
        return v.item()

    def _backprop(self, node, value):
        current = node
        while current is not None:
            current.update(value)
            current = current.parent
            value = -value
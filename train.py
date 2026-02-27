import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque

# 导入之前的模块
from game import TacticalTicTacToe
from model import TacticalZeroNet
from mcts import MCTS
from utils import encode_state

class AlphaZeroTrainer:
    def __init__(self):
        # --- 超参数配置 ---
        self.num_iterations = 100     # 总共训练几大轮 (迭代次数)
        self.num_episodes = 50       # 每一轮自我对弈几局
        self.epochs = 20             # 每次拿到数据后，网络训练几遍
        self.batch_size = 64         # 每次喂给网络多少条数据
        self.lr = 0.001              # 学习率
        self.mcts_sims = 60          # MCTS 每次思考的模拟次数
        self.check_freq = 5          # 每隔几轮保存一次模型
        
        # 初始化模型
        self.nnet = TacticalZeroNet()
        # 如果有 GPU 就用 GPU，没有就 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nnet.to(self.device)
        
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.lr)
        
        # 经验回放池 (保留最近 2000 条数据，防止遗忘)
        self.train_examples_history = deque(maxlen=2000)

    def execute_episode(self):
        """
        执行一局自我对弈，返回这局棋的所有训练数据
        数据格式: [(board_state, mcts_probs, winner_z), ...]
        """
        train_examples = []
        game = TacticalTicTacToe()
        mcts = MCTS(self.nnet, n_simulations=self.mcts_sims)
        
        step = 0
        while True:
            step += 1
            # 1. MCTS 搜索获取概率 (pi)
            # 注意：这里我们需要修改一下 mcts.search，让它返回概率分布，而不是直接返回动作
            # 为了不改动 mcts.py，我们这里手动调用一下 mcts 内部逻辑或者我们假设 search 返回了概率
            # 简单起见，我们重新实例化 MCTS 或者让 MCTS 暴露根节点的计数
            
            # --- 手动运行 MCTS ---
            # 因为之前的 mcts.search 封装太死，这里我们拆解一下逻辑
            root = mcts.search_and_return_root(game) # 需要去 mcts.py稍微改一下接口
            
            # 计算概率 pi (根据访问次数 N)
            counts = [root.children.get(a, 0).n_visits if a in root.children else 0 for a in range(18)]
            sum_counts = sum(counts)
            if sum_counts == 0: # 极其罕见的情况
                pi = [1/18]*18
            else:
                pi = [x / sum_counts for x in counts]
            
            # 2. 保存数据 (State, Pi, CurrentPlayer)
            # 注意：我们还不知道输赢 (z)，所以先存 None，最后回填
            state_tensor = encode_state(game).squeeze(0).numpy() # (7,3,3)
            train_examples.append([state_tensor, pi, game.current_player])
            
            # 3. 选择动作
            # 训练前期要有探索性(按概率随机选)，后期稍微贪婪一点
            action_idx = np.random.choice(len(pi), p=pi)
            
            # 4. 执行动作
            from utils import ActionConverter
            action = ActionConverter.int_to_action(action_idx)
            game.step(action)
            
            if game.game_over:
                # 5. 游戏结束，回填胜负结果 Z
                return_data = []
                for x in train_examples:
                    state_vec, pi_vec, player_at_step = x
                    
                    # 计算 Z: 如果这一步的玩家赢了，Z=+1，输了 Z=-1
                    if game.winner == 0:
                        z = 0
                    elif game.winner == player_at_step:
                        z = 1
                    else:
                        z = -1
                        
                    return_data.append((state_vec, pi_vec, z))
                return return_data

    def train_network(self, examples):
        """
        从经验池中采样，训练神经网络
        """
        self.nnet.train()
        
        for _ in range(self.epochs):
            # 打乱数据
            random.shuffle(examples)
            
            # 小批量训练 (Batch Training)
            for i in range(0, len(examples), self.batch_size):
                batch = examples[i : i + self.batch_size]
                if len(batch) < 4: continue # 数据太少跳过
                
                # 解压数据
                boards, pis, vs = zip(*batch)
                
                # 转为 Tensor
                boards = torch.FloatTensor(np.array(boards)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs)).to(self.device)
                
                # 前向传播
                out_pi, out_v = self.nnet(boards)
                
                # --- 计算 Loss (核心) ---
                # 1. 价值损失 (MSE): (预测赢面 - 真实赢面)^2
                loss_v = F.mse_loss(out_v.view(-1), target_vs)
                
                # 2. 策略损失 (Cross Entropy): 越接近 MCTS 的概率越好
                # 神经网络输出的是 LogSoftmax，所以直接乘目标概率求和即可
                loss_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                
                total_loss = loss_v + loss_pi
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
        print(f"训练完成: Loss V={loss_v.item():.4f}, Loss Pi={loss_pi.item():.4f}")

    def learn(self):
        """主训练循环"""
        for i in range(1, self.num_iterations + 1):
            print(f"--- 迭代 {i}/{self.num_iterations} ---")
            
            # 1. 自我对弈收集数据
            new_examples = []
            for eps in range(self.num_episodes):
                new_examples += self.execute_episode()
                print(f"\r对弈进度: {eps+1}/{self.num_episodes}", end="")
            print("")
            
            # 加入总经验池
            self.train_examples_history.extend(new_examples)
            
            # 2. 训练神经网络
            print(f"开始训练，样本池大小: {len(self.train_examples_history)}")
            # 将 deque 转为 list 用于训练
            self.train_network(list(self.train_examples_history))
            
            # 3. 保存模型
            if i % self.check_freq == 0:
                self.save_model(filename=f"checkpoint_{i}.pth")
                self.save_model(filename="best_model.pth")

    def save_model(self, folder="models", filename="checkpoint.pth"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        torch.save(self.nnet.state_dict(), filepath)
        print(f"模型已保存: {filepath}")

if __name__ == "__main__":
    # 为了让代码能跑，我们需要去改一下 mcts.py 的 search 方法
    # 让它返回 root 节点，而不是直接返回动作 (看下面的说明)
    print("开始 Tactical AlphaZero 训练...")
    trainer = AlphaZeroTrainer()
    trainer.learn()
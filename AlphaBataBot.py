import numpy as np

class AlphaBetaBot:
    def __init__(self, depth=10):
        self.search_depth = depth
        self.nodes_searched = 0
        # 内部状态，每次 get_best_move 时动态更新
        self.my_id = 0
        self.opp_id = 0

    def get_best_move(self, env):
        """
        主接口：根据 env.current_player 自动识别自己是谁。
        保持接口不变，外部只需调用 get_best_move(env)。
        """
        self.nodes_searched = 0
        
        # --- 关键修改：动态识别身份 ---
        # 如果当前轮到 1 走，那我就是 1；如果轮到 -1 走，那我就是 -1
        self.my_id = env.current_player
        self.opp_id = -1 * self.my_id 

        # 获取所有可行步
        actions = env.get_valid_actions()
        if not actions:
            return None

        best_val = -float('inf')
        best_action = None

        # 根节点：永远是“我”在做决策，所以我试图最大化分数 (is_maximizing=True)
        for action in actions:
            next_env = env.clone()
            next_env.step(action)
            
            # 递归搜索：下一层轮到对手走，所以 is_maximizing=False
            val = self._alpha_beta(next_env, self.search_depth - 1, -float('inf'), float('inf'), False)
            
            # print(f"  [Move] {action['type']} {action['pos']} -> Score {val:.1f}")

            if val > best_val:
                best_val = val
                best_action = action

        print(f"Bot({self.my_id}) 搜索完成. 节点数: {self.nodes_searched}, 评分: {best_val:.1f}")
        return best_action

    def _alpha_beta(self, env, depth, alpha, beta, is_maximizing):
        self.nodes_searched += 1

        if depth == 0 or env.game_over:
            return self._evaluate(env)

        actions = env.get_valid_actions()
        if not actions: 
            return self._evaluate(env)

        if is_maximizing:
            # 轮到“我” (my_id) 走 -> 找最大分
            max_eval = -float('inf')
            for action in actions:
                next_env = env.clone()
                next_env.step(action)
                eval_val = self._alpha_beta(next_env, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break 
            return max_eval
        else:
            # 轮到“对手” (opp_id) 走 -> 找最小分 (对手会选对我最不利的)
            min_eval = float('inf')
            for action in actions:
                next_env = env.clone()
                next_env.step(action)
                eval_val = self._alpha_beta(next_env, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break 
            return min_eval

    def _evaluate(self, env):
        """
        通用评估函数。
        正分 -> 对 self.my_id 有利
        负分 -> 对 self.opp_id 有利
        """
        # --- A. 终局评分 ---
        if env.game_over:
            if env.winner == self.my_id:
                # 我赢了
                return 10000.0 - env.costs[self.my_id]
            elif env.winner == self.opp_id:
                # 敌人赢了
                return -10000.0 + env.costs[self.opp_id]
            else:
                return 0.0 # 平局

        # --- B. 过程评分 (启发式) ---
        score = 0.0

        # 1. 武器差 (我的武器 - 敌人的武器)
        # 无论我是 1 还是 -1，这个公式都成立
        score += 50.0 * (env.weapons[self.my_id] - env.weapons[self.opp_id])

        # 2. 关键路线评分
        lines = []
        lines.extend([env.board[:, i] for i in range(3)]) # 列
        lines.append(env.board.diagonal())                # 对角1
        lines.append(np.fliplr(env.board).diagonal())     # 对角2

        for line in lines:
            cnt_my = np.sum(line == self.my_id)   # 我的棋子数
            cnt_opp = np.sum(line == self.opp_id) # 敌人的棋子数

            # 进攻机会 (这行没有敌人的子)
            if cnt_opp == 0:
                if cnt_my == 2: score += 150.0 # 我差一步连成
                elif cnt_my == 1: score += 20.0

            # 防守威胁 (这行没有我的子)
            if cnt_my == 0:
                if cnt_opp == 2: score -= 150.0 # 敌人差一步连成 (危险！)
                elif cnt_opp == 1: score -= 20.0

        # 3. 成本优势 (敌人花钱多是对我有理，我花钱少也是对我有理)
        score += (env.costs[self.opp_id] - env.costs[self.my_id]) * 1.0

        return score
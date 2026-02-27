import numpy as np

class TacticalTicTacToe:
    # --- 静态配置 ---
    TIME_COST_ROWS = [8.0, 5.0, 3.0] 
    TIME_COST_ATTACK = 3
    P_FAIL_PLACE = 0
    P_FAIL_ATTACK = 0
    MAX_COST = 100.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.weapons = {1: 3, -1: 3}
        self.costs = {1: 0.0, -1: 0.0}
        self.current_player = 1 
        self.winner = None
        self.game_over = False
        return self

    def clone(self):
        new_env = TacticalTicTacToe()
        new_env.board = self.board.copy()
        new_env.weapons = self.weapons.copy()
        new_env.costs = self.costs.copy()
        new_env.current_player = self.current_player
        new_env.winner = self.winner
        new_env.game_over = self.game_over
        return new_env

    @staticmethod
    def get_expected_cost(action_type, row_idx=0):
        if action_type == 'ATTACK':
            return TacticalTicTacToe.TIME_COST_ATTACK / (1.0 - TacticalTicTacToe.P_FAIL_ATTACK)
        else:
            r = max(0, min(row_idx, 2))
            base = TacticalTicTacToe.TIME_COST_ROWS[r]
            return base / (1.0 - TacticalTicTacToe.P_FAIL_PLACE)

    def get_valid_actions(self):
        if self.game_over: return []
        actions = []
        
        # 1. 放置
        empty_spots = np.argwhere(self.board == 0)
        for r, c in empty_spots:
            actions.append({'type': 'PLACE', 'pos': (r, c)})
            
        # 2. 攻击
        p = int(self.current_player)
        if self.weapons[p] > 0:
            enemy_spots = np.argwhere(self.board == -p)
            for r, c in enemy_spots:
                actions.append({'type': 'ATTACK', 'pos': (r, c)})
        return actions

    def step(self, action):
        if self.game_over: return

        act_type = action['type']
        r, c = action['pos']
        p = int(self.current_player)
        
        cost = 0.0
        if act_type == 'PLACE':
            self.board[r, c] = p
            cost = self.get_expected_cost('PLACE', r)
        elif act_type == 'ATTACK':
            self.board[r, c] = 0
            self.weapons[p] -= 1
            cost = self.get_expected_cost('ATTACK')

        self.costs[p] += cost
        self._check_game_status()
        
        if not self.game_over:
            self.current_player *= -1

    def _check_game_status(self):
        # 1. 胜利判定
        for p in [1, -1]:
            col_win = np.any(np.all(self.board == p, axis=0))
            diag1 = np.all(self.board.diagonal() == p)
            diag2 = np.all(np.fliplr(self.board).diagonal() == p)
            
            if col_win or diag1 or diag2:
                self.winner = p
                self.game_over = True
                return

        # 2. 成本超标
        if self.costs[1] > self.MAX_COST or self.costs[-1] > self.MAX_COST:
            self.winner = 0 
            self.game_over = True
            return

        # 3. 死局判定
        if not np.any(self.board == 0):
            self.winner = 0
            self.game_over = True
            return
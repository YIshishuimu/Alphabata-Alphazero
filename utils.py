import numpy as np
import torch

class ActionConverter:
    @staticmethod
    def int_to_action(action_idx):
        if action_idx < 9:
            r, c = action_idx // 3, action_idx % 3
            return {'type': 'PLACE', 'pos': (r, c)}
        else:
            idx = action_idx - 9
            r, c = idx // 3, idx % 3
            return {'type': 'ATTACK', 'pos': (r, c)}

    @staticmethod
    def action_to_int(action):
        r, c = action['pos']
        base = 0 if action['type'] == 'PLACE' else 9
        return base + r * 3 + c

def encode_state(game_env):
    """将环境转换为 (1, 7, 3, 3) Tensor"""
    board = game_env.board
    me = game_env.current_player
    enemy = -me
    
    layer_my_pieces = (board == me).astype(np.float32)
    layer_enemy_pieces = (board == enemy).astype(np.float32)
    
    fill = lambda val: np.full((3, 3), val, dtype=np.float32)
    layer_my_ammo = fill(game_env.weapons[me] / 3.0)
    layer_enemy_ammo = fill(game_env.weapons[enemy] / 3.0)
    layer_my_cost = fill(game_env.costs[me] / game_env.MAX_COST)
    layer_enemy_cost = fill(game_env.costs[enemy] / game_env.MAX_COST)
    
    layer_row_costs = np.array([
        [1.0, 1.0, 1.0], [0.6, 0.6, 0.6], [0.3, 0.3, 0.3]
    ], dtype=np.float32)
    
    state = np.stack([
        layer_my_pieces, layer_enemy_pieces,
        layer_my_ammo, layer_enemy_ammo,
        layer_my_cost, layer_enemy_cost,
        layer_row_costs
    ])
    return torch.tensor(state).unsqueeze(0)

def get_valid_moves_mask(game_env):
    mask = np.zeros(18, dtype=np.float32)
    valid_actions = game_env.get_valid_actions()
    for action in valid_actions:
        idx = ActionConverter.action_to_int(action)
        mask[idx] = 1.0
    return torch.tensor(mask)
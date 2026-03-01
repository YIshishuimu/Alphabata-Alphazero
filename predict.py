import numpy as np
import re
from game import TacticalTicTacToe
from AlphaBataBot import AlphaBetaBot

def parse_state_file(filename):
    """解析 txt 文件获取局面信息"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析棋盘 (提取 board: 下方的三行数字)
    board_match = re.search(r'board:\s*([\s\S]*?)\n\n', content + '\n\n')
    board_str = board_match.group(1).strip()
    board = [list(map(int, row.split(','))) for row in board_str.split('\n')]

    # 解析武器
    weapons = {}
    weapons[1] = int(re.search(r'weapons:[\s\S]*?1:\s*(\d+)', content).group(1))
    weapons[-1] = int(re.search(r'weapons:[\s\S]*?-1:\s*(\d+)', content).group(1))

    # 解析成本
    costs = {}
    costs[1] = float(re.search(r'costs:[\s\S]*?1:\s*([\d.]+)', content).group(1))
    costs[-1] = float(re.search(r'costs:[\s\S]*?-1:\s*([\d.]+)', content).group(1))

    # 解析下一位玩家
    next_player = int(re.search(r'next_player:\s*(-?\d+)', content).group(1))

    return board, weapons, costs, next_player

def run_prediction():
    try:
        # 1. 从 TXT 读取数据
        board_data, weapons, costs, next_player = parse_state_file('state.txt')
        
        # 2. 注入环境
        env = TacticalTicTacToe()
        env.board = np.array(board_data, dtype=np.int8)
        env.weapons = weapons
        env.costs = costs
        env.current_player = next_player
        
        # 3. AI 分析
        bot = AlphaBetaBot(depth=10)
        
        print("="*40)
        print(f"读取文件成功！分析目标: {'玩家 1 (○)' if next_player == 1 else '玩家 2 (×)'}")
        print(f"当前状态 -> 成本: {costs} | 武器: {weapons}")
        print("-" * 40)
        
        best_action = bot.get_best_move(env)
        
        if best_action:
            print("="*40)
            print(f"💡 建议操作: {best_action['type']}")
            print(f"📍 目标位置: {best_action['pos']}")
            print("="*40)
        else:
            print("无法找到合法移动。")

    except Exception as e:
        print(f"读取或解析文件失败: {e}")

if __name__ == "__main__":
    run_prediction()
import torch
import sys

# 导入我们之前写的模块
from game import TacticalTicTacToe
from model import TacticalZeroNet
from mcts import MCTS
# 虽然 utils 里的函数在 main 里没直接调用，但保留导入也没坏处
from utils import encode_state, get_valid_moves_mask, ActionConverter

if __name__ == "__main__":
    print(">>> 初始化 Tactical AlphaZero 演示系统...")

    # 1. 初始化游戏环境 (您漏掉的就是这一行)
    game = TacticalTicTacToe()
    
    # 2. 初始化神经网络
    net = TacticalZeroNet()
    
    # 3. 尝试加载训练好的模型
    model_path = "models/best_model.pth"
    try:
        # map_location='cpu' 保证即使没显卡也能跑
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
        net.eval() # 切换到预测模式 (冻结 BatchNorm 等层)
        print(f">>> 成功加载模型: {model_path}")
    except FileNotFoundError:
        print(">>> 警告：未找到模型文件！AI 将使用随机参数（智障模式）。")
        print(">>> 请先运行 python train.py 进行训练。")
    except Exception as e:
        print(f">>> 加载模型出错: {e}")

    # 4. 初始化 MCTS
    # n_simulations 越大，AI 越强，但思考时间越长
    # 演示时设为 100-200 比较合适
    mcts = MCTS(net, n_simulations=100)

    print("\n>>> 游戏开始！(AI 左右互搏演示)")
    print("图例: 1 = 先手AI (蓝), -1 = 后手AI (红), 0 = 空")
    
    step_count = 0
    
    # --- 游戏主循环 ---
    while not game.game_over:
        player_name = "AI-1 (先手)" if game.current_player == 1 else "AI-2 (后手)"
        print(f"\n--- 回合 {step_count} [{player_name}] ---")
        
        # 1. MCTS 思考最佳动作
        # 注意：这里我们让两个 AI 共用同一个大脑 (net) 和同一套 MCTS
        # 这就是 AlphaZero 的"自我对弈"逻辑
        print("思考中...", end="", flush=True)
        action = mcts.search(game)
        print(" 完成!")
        
        # 2. 打印决策结果
        act_type = action['type']
        pos = action['pos']
        print(f"AI 决定: {act_type} -> 位置 {pos}")
        
        # 3. 在真实棋盘上执行
        game.step(action)
        
        # 4. 打印当前局势
        print(game.board)
        print(f"成本: P1={game.costs[1]:.1f}, P2={game.costs[-1]:.1f}")
        print(f"弹药: P1={game.weapons[1]}, P2={game.weapons[-1]}")
        
        step_count += 1
        
        # (可选) 稍微暂停一下，方便人类观察，不然刷屏太快
        # import time; time.sleep(1)

    # --- 游戏结束结算 ---
    print("\n" + "="*30)
    print(">>> 游戏结束!")
    print("="*30)
    
    if game.winner == 0:
        print("结果: 平局 (双方僵持 或 成本耗尽)")
    else:
        winner_name = "AI-1 (先手)" if game.winner == 1 else "AI-2 (后手)"
        print(f"结果: 玩家 {game.winner} [{winner_name}] 获胜!")
        
    print(f"最终成本: P1={game.costs[1]:.1f}, P2={game.costs[-1]:.1f}")
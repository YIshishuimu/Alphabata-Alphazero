import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import random
import contextlib
import torch

# ==========================================
# 1. 导入核心模块
# (必须存在之前写的 game.py, model.py, mcts.py)
# ==========================================
try:
    from game import TacticalTicTacToe
    from model import TacticalZeroNet
    from mcts import MCTS
    from utils import ActionConverter
    from AlphaBataBot import AlphaBetaBot
except ImportError:
    print("❌ 错误: 缺少核心文件！请确保 game.py, model.py, mcts.py 在同一目录下。")
    exit()

# 全局配置
TIME_SCALE = 0.1  # 游戏速度倍率 (越小越快)

# ==========================================
# 2. 定义 Bot 包装器
# ==========================================

class AlphaZeroBot:
    """
    AlphaZero 包装器：加载训练好的神经网络 + MCTS
    """
    def __init__(self, model_path="models/best_model.pth", simulations=200):
        self.device = torch.device("cpu") # 推理通常 CPU 就够了，避免抢占训练资源
        self.net = TacticalZeroNet().to(self.device)
        
        try:
            # 加载模型
            state_dict = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(state_dict)
            self.net.eval()
            print(f"✅ AlphaZero 模型加载成功: {model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败，将使用随机参数: {e}")

        # 初始化 MCTS 引擎
        self.simulations = simulations
        self.mcts = MCTS(self.net, n_simulations=self.simulations)

    def get_best_move(self, env):
        # MCTS 需要一个全新的环境对象来模拟，env 是当前局面的快照
        # search 会返回最佳动作的字典 {'type':..., 'pos':...}
        return self.mcts.search(env)

# ==========================================
# 3. 游戏控制器 (Game Controller)
# ==========================================

class GameController:
    def __init__(self):
        # 使用真实的逻辑环境
        self.env = TacticalTicTacToe()
        
        self.lock = threading.Lock()
        self.logic_lock = threading.Lock()
        self.running = True
        self.log_msgs = []
        self.unlock_times = {1: 0.0, -1: 0.0}
        self.is_paused = False
        self.pause_start_time = 0.0

        # 随机开局
        all_coords = [(r, c) for r in range(3) for c in range(3)]
        block_count = 1 
        chosen_spots = random.sample(all_coords, block_count)
        
        init_log = f"=== 初始化 (预设: {block_count}) ===\n"
        for r, c in chosen_spots:
            owner = random.choice([1, -1])
            self.env.board[r, c] = owner
            init_log += f"({r}, {c}) -> {owner}\n"
            
        self.log(init_log)
        self.log("=== 对抗开始: AlphaZero vs AlphaBeta ===")

    def log(self, msg):
        print(msg)
        self.log_msgs.append(msg)
        if len(self.log_msgs) > 12:
            self.log_msgs.pop(0)

    @contextlib.contextmanager
    def freeze_time(self):
        self.logic_lock.acquire()
        t_start = time.time()
        with self.lock:
            self.is_paused = True
            self.pause_start_time = t_start
        try:
            yield
        finally:
            t_end = time.time()
            duration = t_end - t_start
            with self.lock:
                self.is_paused = False
                self.unlock_times[1] += duration
                self.unlock_times[-1] += duration
            self.logic_lock.release()

    def get_display_data(self):
        now = time.time()
        with self.lock:
            ref_time = self.pause_start_time if self.is_paused else now
            cd1 = max(0.0, self.unlock_times[1] - ref_time)
            cd2 = max(0.0, self.unlock_times[-1] - ref_time)
            return {
                'board': self.env.board.copy(),
                'costs': self.env.costs.copy(),
                'weapons': self.env.weapons.copy(),
                'game_over': self.env.game_over,
                'winner': self.env.winner,
                'logs': list(self.log_msgs),
                'cooldowns': {1: cd1, -1: cd2},
                'is_paused': self.is_paused 
            }

    def can_move(self, player_id):
        return time.time() >= self.unlock_times[player_id]

    def execute_move(self, player_id, action):
        with self.lock:
            if self.env.game_over: return False, 0
            
            # 计算耗时 (从 Game 类静态方法获取)
            cost = 0.0
            if action['type'] == 'PLACE':
                cost = TacticalTicTacToe.get_expected_cost('PLACE', action['pos'][0])
            else:
                cost = TacticalTicTacToe.get_expected_cost('ATTACK')

            # 验证与执行
            self.env.current_player = player_id
            
            # 简单的合法性检查 (严格检查在 env.step 内部也有，这里为了防止报错)
            r, c = action['pos']
            if action['type'] == 'PLACE' and self.env.board[r, c] != 0: return False, 0
            if action['type'] == 'ATTACK' and self.env.board[r, c] != -player_id: return False, 0
            
            self.env.step(action)
            
            real_duration = cost * TIME_SCALE
            self.unlock_times[player_id] = time.time() + real_duration
            return True, real_duration

# ==========================================
# 4. Bot 线程逻辑 (核心对抗)
# ==========================================

def run_bot_thread(controller, player_id, bot_type):
    # 初始化 AI
    ai = None
    name = ""
    
    if bot_type == 'AlphaZero':
        ai = AlphaZeroBot(simulations=200) # 搜索次数越多越强
        name = "AlphaZero (P1)" if player_id == 1 else "AlphaZero (P2)"
    elif bot_type == 'AlphaBeta':
        ai = AlphaBetaBot(depth=6)         # 深度越深越强，但越慢
        name = "AlphaBeta (P1)" if player_id == 1 else "AlphaBeta (P2)"

    controller.log(f"🧠 {name} 已就绪")

    while controller.running:
        time.sleep(0.05) 
        
        # 1. 检查物理冷却
        if not controller.can_move(player_id): continue

        # 2. 尝试抢占时间静止锁
        action = None
        success = False
        duration = 0.0

        try:
            with controller.freeze_time():
                if controller.env.game_over: break
                
                # 获取环境快照
                with controller.lock:
                    env_snapshot = controller.env.clone()
                
                # 确保当前操作者是自己 (对于 Minimax 这种需要区分视角的很重要)
                env_snapshot.current_player = player_id

                # AI 思考
                # 注意：如果我是 P2(-1)，且算法是只懂 P1 视角的，这里可能需要 flip_perspective
                # 但我们的 MCTS 和 Minimax 写法已经兼容了 P1/P2，不需要翻转
                try:
                    action = ai.get_best_move(env_snapshot)
                except Exception as e:
                    print(f"AI Error: {e}")

                # 执行动作
                if action:
                    success, duration = controller.execute_move(player_id, action)
            
            # --- 退出时间静止 ---
            
            if action and success:
                controller.log(f"{name} -> {action['type']} {action['pos']}")
                # 物理等待 (让出 CPU)
                time.sleep(duration)
            else:
                time.sleep(0.1) # 没想出来或无效动作，休息一下

        except Exception as e:
            print(f"Critical Error in {name}: {e}")
            break

# ==========================================
# 5. GUI 界面 (复用之前的逻辑)
# ==========================================

class BattleGUI:
    def __init__(self, root, controller):
        self.root = root
        self.ctrl = controller
        self.root.title("Tactical Arena: AlphaZero vs AlphaBeta")
        self.root.geometry("640x650")

        self.colors = {0: "#F0F0F0", 1: "#3399FF", -1: "#FF6666"}
        self.symbols = {0: "", 1: "○", -1: "×"}
        
        self._setup_ui()
        self.update_ui()

    def _setup_ui(self):
        # 顶部栏
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(fill="x")

        # P1 左侧
        f_p1 = tk.Frame(top_frame, width=220)
        f_p1.pack(side="left", padx=20)
        tk.Label(f_p1, text="AlphaZero (P1)", fg="#3399FF", font=("Arial", 14, "bold")).pack()
        self.bar_p1 = ttk.Progressbar(f_p1, length=140, mode='determinate')
        self.bar_p1.pack(pady=5)
        self.status_p1 = tk.Label(f_p1, text="Ready", fg="gray")
        self.status_p1.pack()

        # VS
        tk.Label(top_frame, text="VS", font=("Impact", 20)).pack(side="left")

        # P2 右侧
        f_p2 = tk.Frame(top_frame, width=220)
        f_p2.pack(side="right", padx=20)
        tk.Label(f_p2, text="AlphaBeta (P2)", fg="#FF6666", font=("Arial", 14, "bold")).pack()
        self.bar_p2 = ttk.Progressbar(f_p2, length=140, mode='determinate')
        self.bar_p2.pack(pady=5)
        self.status_p2 = tk.Label(f_p2, text="Ready", fg="gray")
        self.status_p2.pack()

        # 棋盘
        board_frame = tk.Frame(self.root, pady=10)
        board_frame.pack()
        self.cells = [[None]*3 for _ in range(3)]
        for r in range(3):
            for c in range(3):
                lbl = tk.Label(board_frame, text="", font=("Arial", 28, "bold"), 
                               width=6, height=3, relief="groove", bg="#F0F0F0")
                lbl.grid(row=r, column=c, padx=4, pady=4)
                self.cells[r][c] = lbl

        # 数据栏
        self.stats_lbl = tk.Label(self.root, text="Waiting...", font=("Consolas", 11))
        self.stats_lbl.pack(pady=10)

        # 日志
        self.log_text = tk.Text(self.root, height=8, bg="#333", fg="#EEE", state="disabled")
        self.log_text.pack(fill="x", padx=10, pady=5)

    def update_ui(self):
        if not self.ctrl.running: return

        data = self.ctrl.get_display_data()
        
        # 1. 棋盘
        for r in range(3):
            for c in range(3):
                owner = data['board'][r, c]
                self.cells[r][c].config(text=self.symbols[owner], bg=self.colors[owner], fg="white" if owner!=0 else "black")

        # 2. 状态与进度
        paused = data['is_paused']
        
        # Helper func
        def update_player_status(cd, bar, lbl, name):
            if cd > 0:
                if paused:
                    lbl.config(text=f"Time Stop...", fg="#999")
                else:
                    lbl.config(text=f"Cooling: {cd:.1f}s", fg="#E67E22")
                bar['value'] = 100
            else:
                if paused:
                    lbl.config(text=f"Thinking...", fg="#9B59B6") # 紫色表示正在思考
                else:
                    lbl.config(text="READY", fg="#27AE60")
                bar['value'] = 0

        update_player_status(data['cooldowns'][1], self.bar_p1, self.status_p1, "P1")
        update_player_status(data['cooldowns'][-1], self.bar_p2, self.status_p2, "P2")

        # 3. 数据
        c1, c2 = data['costs'][1], data['costs'][-1]
        w1, w2 = data['weapons'][1], data['weapons'][-1]
        
        info = f"COST: {c1:.0f} vs {c2:.0f}  |  AMMO: {w1} vs {w2}"
        if data['game_over']:
            w = data['winner']
            win_txt = "AlphaZero WINS!" if w==1 else ("AlphaBeta WINS!" if w==-1 else "DRAW")
            info = f"[{win_txt}]  {info}"
            if self.ctrl.running:
                self.ctrl.running = False
                messagebox.showinfo("Result", win_txt)
        
        self.stats_lbl.config(text=info)

        # 4. 日志
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        for msg in data['logs']:
            self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

        self.root.after(50, self.update_ui)

# ==========================================
# 启动
# ==========================================

if __name__ == "__main__":
    ctrl = GameController()
    
    # --- 选手配置 ---
    # Player 1: AlphaZero (加载训练模型)
    t1 = threading.Thread(target=run_bot_thread, args=(ctrl, 1, 'AlphaZero'))
    t1.daemon = True
    t1.start()
    
    # Player 2: AlphaBeta (传统强算法)
    t2 = threading.Thread(target=run_bot_thread, args=(ctrl, -1, 'AlphaBeta'))
    t2.daemon = True
    t2.start()
    
    root = tk.Tk()
    app = BattleGUI(root, ctrl)
    root.mainloop()
    
    ctrl.running = False
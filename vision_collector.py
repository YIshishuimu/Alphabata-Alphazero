import cv2
import numpy as np

# --- 配置 ---
STATE_FILE = "state.txt"
MIN_BOARD_AREA = 30000 

# HSV 颜色阈值 (针对红蓝立方体棋子)
LOWER_BLUE, UPPER_BLUE = np.array([100, 80, 40]), np.array([135, 255, 255])
LOWER_RED1, UPPER_RED1 = np.array([0, 80, 40]), np.array([10, 255, 255])
LOWER_RED2, UPPER_RED2 = np.array([170, 80, 40]), np.array([180, 255, 255])

def save_to_txt(board):
    """仅更新棋盘部分，保持其他默认值，或根据需要手动修改此处的武器和成本"""
    # 默认假设：武器3个，成本0，当前玩家1
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        f.write("# 视觉自动采集结果\nboard:\n")
        for row in board:
            f.write(", ".join(map(str, row)) + "\n")
        f.write("\nweapons:\n1: 3\n-1: 3\n\ncosts:\n1: 0.0\n-1: 0.0\n")
        f.write("\nnext_player: 1\n")

def get_logic_board(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    board = np.zeros((3, 3), dtype=int)
    h, w, _ = warped.shape
    ch, cw = h // 3, w // 3
    for r in range(3):
        for c in range(3):
            # 取格子中心区域，避开黑色镂空框架线干扰
            roi = hsv[r*ch + ch//5 : (r+1)*ch - ch//5, c*cw + cw//5 : (c+1)*cw - cw//5]
            mask_b = cv2.inRange(roi, LOWER_BLUE, UPPER_BLUE)
            mask_r = cv2.inRange(roi, LOWER_RED1, UPPER_RED1) + cv2.inRange(roi, LOWER_RED2, UPPER_RED2)
            if cv2.countNonZero(mask_b) > (roi.shape[0] * roi.shape[1] * 0.15):
                board[r, c] = 1
            elif cv2.countNonZero(mask_r) > (roi.shape[0] * roi.shape[1] * 0.15):
                board[r, c] = -1
    return board

def main():
    cap = cv2.VideoCapture(0)
    print("📷 视觉采集开启... 自动识别黑色镂空棋盘中")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 针对黑色框架的预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        board_cnt = None
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(cnt) > MIN_BOARD_AREA:
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                if len(approx) == 4:
                    board_cnt = approx
                    break

        if board_cnt is not None:
            cv2.drawContours(frame, [board_cnt], -1, (0, 255, 0), 2)
            # 排序点并变换
            pts = board_cnt.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
            
            M = cv2.getPerspectiveTransform(rect, np.float32([[0,0],[300,0],[300,300],[0,300]]))
            warped = cv2.warpPerspective(frame, M, (300, 300))
            
            logic_board = get_logic_board(warped)
            cv2.imshow("Board_Cropped", warped)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'): # 按 S 键保存数据到 state.txt
                save_to_txt(logic_board)
                print(f"✅ 棋盘数据已写入 {STATE_FILE} :\n{logic_board}")
        
        cv2.imshow("Vision Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
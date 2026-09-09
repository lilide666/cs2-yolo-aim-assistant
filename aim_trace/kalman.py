import numpy as np


class KalmanPredictor:
    """
    卡尔曼滤波轨迹预测器。
    状态向量: [x, y, vx, vy]   (位置 + 速度)
    测量向量: [x, y]
    用速度外推预测目标在"未来 lead_seconds 秒"后的位置，补偿移动延迟。
    """
    def __init__(self, process_noise=1e-2, measure_noise=2e1, dt=1.0):
        self.dt = dt
        # 状态转移矩阵 F: 位置 += 速度*dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        # 测量矩阵 H: 只观测位置
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        # 过程噪声协方差 Q
        q = process_noise
        self.Q = np.eye(4, dtype=np.float32) * q
        self.Q[2, 2] *= 10.0   # 速度分量方差更大，允许速度变化
        self.Q[3, 3] *= 10.0
        # 测量噪声协方差 R
        self.R = np.eye(2, dtype=np.float32) * measure_noise
        # 误差协方差 P
        self.P = np.eye(4, dtype=np.float32) * 100.0
        # 状态向量
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.initialized = False

    def update(self, mx, my):
        """输入测量位置，返回滤波后的位置 (fx, fy)。"""
        z = np.array([[mx], [my]], dtype=np.float32)

        if not self.initialized:
            self.x[0, 0] = mx
            self.x[1, 0] = my
            self.initialized = True
            return mx, my

        # 预测
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # 更新
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        return float(self.x[0, 0]), float(self.x[1, 0])

    def predict(self, lead_seconds):
        """外推 lead_seconds 秒后的位置 (px, py)。"""
        px = float(self.x[0, 0]) + float(self.x[2, 0]) * lead_seconds
        py = float(self.x[1, 0]) + float(self.x[3, 0]) * lead_seconds
        return px, py

    @property
    def filtered_pos(self):
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self):
        return float(self.x[2, 0]), float(self.x[3, 0])

    def reset(self):
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 100.0
        self.initialized = False
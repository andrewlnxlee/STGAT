import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import cholesky

# ===================================================================
# 1. 核心组件: UKF (无迹卡尔曼滤波器) 与 CTRV 运动模型
# ===================================================================

N_DIM = 5; ALPHA = 0.1; BETA = 2.0; KAPPA = 0.0
LAMBDA = ALPHA**2 * (N_DIM + KAPPA) - N_DIM

def ctrv_motion_model(x_state, dt=1.0):
    x, y, v, yaw, yaw_rate = x_state
    if abs(yaw_rate) > 1e-4:
        nx = x + (v / yaw_rate) * (np.sin(yaw + yaw_rate * dt) - np.sin(yaw))
        ny = y + (v / yaw_rate) * (np.cos(yaw) - np.cos(yaw + yaw_rate * dt))
        nyaw = (yaw + yaw_rate * dt + np.pi) % (2 * np.pi) - np.pi
    else:
        nx = x + v * dt * np.cos(yaw)
        ny = y + v * dt * np.sin(yaw)
        nyaw = yaw
    return np.array([nx, ny, v, nyaw, yaw_rate])

class UnscentedKalmanFilter:
    def __init__(self, initial_state, initial_cov):
        self.x = initial_state
        self.P = initial_cov
        
        # --- 关键修改 1: 增大测量噪声 R ---
        # GNN的质心会因群形态变化而抖动，R太小会让滤波器死跟抖动，导致速度估计不稳。
        self.R = np.diag([25.0, 25.0]) 

        # --- 关键修改 2: 大幅增加过程噪声 Q ---
        # 告诉滤波器：要假设目标随时可能机动！不要过于相信自己的预测模型。
        # 显著增加了速度(v)和转弯率(yaw_rate)的不确定性。
        self.Q = np.diag([2.0, 2.0, 10.0, 0.5, 0.5]) 
        
        self.wm = np.full(2 * N_DIM + 1, 0.5 / (N_DIM + LAMBDA))
        self.wc = self.wm.copy()
        self.wm[0] = LAMBDA / (N_DIM + LAMBDA)
        self.wc[0] = self.wm[0] + (1 - ALPHA**2 + BETA)

    def _generate_sigma_points(self):
        points = np.zeros((2 * N_DIM + 1, N_DIM))
        try:
            L = cholesky((N_DIM + LAMBDA) * self.P)
        except np.linalg.LinAlgError:
            L = np.sqrt(N_DIM + LAMBDA) * np.sqrt(np.abs(np.diag(self.P))) * np.eye(N_DIM)
        points[0] = self.x
        for i in range(N_DIM):
            points[i + 1] = self.x + L[i, :]
            points[N_DIM + 1 + i] = self.x - L[i, :]
        return points

    def predict(self, dt=1.0):
        sigma_points = self._generate_sigma_points()
        propagated_points = np.array([ctrv_motion_model(s, dt) for s in sigma_points])
        self.x = np.sum(self.wm[:, np.newaxis] * propagated_points, axis=0)
        self.x[3] = (self.x[3] + np.pi) % (2 * np.pi) - np.pi
        
        pred_P = np.zeros((N_DIM, N_DIM))
        for k in range(2 * N_DIM + 1):
            diff = propagated_points[k] - self.x
            diff[3] = (diff[3] + np.pi) % (2 * np.pi) - np.pi
            pred_P += self.wc[k] * np.outer(diff, diff)
        self.P = pred_P + self.Q

    def update(self, z):
        sigma_points = self._generate_sigma_points()
        meas_points = sigma_points[:, :2]
        z_pred = np.sum(self.wm[:, np.newaxis] * meas_points, axis=0)
        
        S = np.zeros((2, 2))
        P_xz = np.zeros((N_DIM, 2))
        for k in range(2 * N_DIM + 1):
            z_diff = meas_points[k] - z_pred
            S += self.wc[k] * np.outer(z_diff, z_diff)
            x_diff = sigma_points[k] - self.x
            x_diff[3] = (x_diff[3] + np.pi) % (2 * np.pi) - np.pi
            P_xz += self.wc[k] * np.outer(x_diff, z_diff)
        S += self.R
        
        try:
            S_inv = np.linalg.inv(S)
            K = P_xz @ S_inv
            self.x += K @ (z - z_pred)
            self.P -= K @ S @ K.T
            self.x[3] = (self.x[3] + np.pi) % (2 * np.pi) - np.pi
        except np.linalg.LinAlgError:
            return

# ===================================================================
# 2. GNN 后处理器主类 (使用UKF) - 优化版
# ===================================================================
class GNNPostProcessorUKF:
    def __init__(self, dist_thresh=9.0): # --- 关键修改 3: 放宽匹配门限 ---
        self.tracks = {}
        self.next_id = 1
        self.max_age = 5
        self.dist_thresh = dist_thresh

    def _mahalanobis_distance(self, ukf_filter, z):
        x_pred, P_pred = ukf_filter.x, ukf_filter.P
        z_pred = x_pred[:2]
        H = np.array([[1,0,0,0,0], [0,1,0,0,0]])
        S = H @ P_pred @ H.T + ukf_filter.R
        try:
            S_inv = np.linalg.inv(S)
            diff = z - z_pred
            return np.sqrt(diff.T @ S_inv @ diff)
        except np.linalg.LinAlgError:
            return float('inf')

    def update(self, detected_centers):
        for trk in self.tracks.values():
            trk['ukf'].predict(dt=1.0)
            trk['age'] += 1

        active_ids = list(self.tracks.keys())
        assignment, used_dets = {}, set()

        if active_ids and detected_centers:
            cost = np.array([[self._mahalanobis_distance(self.tracks[tid]['ukf'], det) 
                              for det in detected_centers] for tid in active_ids])
            row, col = linear_sum_assignment(cost)
            
            for r_i, c_i in zip(row, col):
                if cost[r_i, c_i] < self.dist_thresh:
                    tid = active_ids[r_i]
                    trk = self.tracks[tid]
                    det_pos = detected_centers[c_i]

                    # --- 关键修改 4: 改进的速度和航向初始化 ---
                    # 如果是刚创建的航迹(is_new)，用第一次的位移来初始化速度和航向
                    if trk.get('is_new', False):
                        dx = det_pos[0] - trk['ukf'].x[0]
                        dy = det_pos[1] - trk['ukf'].x[1]
                        speed = np.sqrt(dx**2 + dy**2)
                        yaw = np.arctan2(dy, dx)
                        trk['ukf'].x[2] = speed
                        trk['ukf'].x[3] = yaw
                        trk['is_new'] = False # 状态已初始化，取消标记
                    
                    trk['ukf'].update(det_pos)
                    trk['age'] = 0
                    current_pos = trk['ukf'].x[:2]
                    trk['trace'].append(current_pos)
                    if len(trk['trace']) > 50: trk['trace'].pop(0)
                    assignment[c_i] = tid
                    used_dets.add(c_i)
        
        for i in range(len(detected_centers)):
            if i not in used_dets:
                self._create_track(self.next_id, detected_centers[i])
                assignment[i] = self.next_id
                self.next_id += 1
                
        self._cleanup_tracks()
        return assignment
        
    def _create_track(self, tid, pos):
        initial_state = np.array([pos[0], pos[1], 0.0, 0.0, 0.0])
        initial_cov = np.diag([25.0, 25.0, 100.0, (np.pi/2)**2, 1.0])
        self.tracks[tid] = {
            'ukf': UnscentedKalmanFilter(initial_state, initial_cov),
            'age': 0, 'trace': [np.array(pos)],
            'is_new': True  # --- 关键修改 4: 标记为新航迹 ---
        }

    def _cleanup_tracks(self):
        to_del = [tid for tid, trk in self.tracks.items() if trk['age'] > self.max_age]
        for tid in to_del: del self.tracks[tid]
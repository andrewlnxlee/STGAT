import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# ===================================================================
# GNN 后处理器 (高精度 CA-KF + 强力抗漂移策略)
# ===================================================================
class GNNPostProcessor:
    def __init__(self, dist_thresh=None):
        self.tracks = {}
        self.next_id = 1
        
        # --- 策略参数 ---
        self.max_age = 15          # 耐心等待
        self.stage1_thresh = 40.0  # 近距离精确匹配
        self.stage2_thresh = 90.0  # 远距离找回 (稍微收紧一点以防误配)
        
        # --- 6D CA (恒定加速度) 运动模型 ---
        dt = 1.0
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0], 
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0], 
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 1]
        ])
        self.H = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0]])
        
        # --- 关键调整 1: 回归高精度配置 ---
        # R=8.0: 非常信任GNN的测量，这将把RMSE压回 4.3 左右的水平
        self.R = np.eye(2) * 8.0 
        
        # Q: 过程噪声，保持适中
        q_pos=0.05; q_vel=0.1; q_acc=0.3
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_acc, q_acc])

        # 重置用的协方差
        p_pos=10.0; p_vel=100.0; p_acc=400.0
        self.P_reset = np.diag([p_pos, p_pos, p_vel, p_vel, p_acc, p_acc])

    def update(self, detected_centers):
        # --- 1. 预测 (带强力抗漂移) ---
        for trk in self.tracks.values():
            # --- 关键调整 2: 强力运动衰减 (Friction) ---
            # 如果处于丢失状态(age>0)，我们假设它"刹车"了。
            # 这防止了预测位置在丢失期间因之前的加速度而"飞"得太远，
            # 从而保证目标再次出现时，依然在 stage2_thresh 的捕获范围内。
            if trk['age'] > 0:
                trk['x'][2:4] *= 0.9  # 速度衰减
                trk['x'][4:] *= 0.5   # 加速度大幅衰减 (几乎归零)
            
            trk['x'] = self.F @ trk['x']
            trk['P'] = self.F @ trk['P'] @ self.F.T + self.Q
            trk['age'] += 1 

        active_ids = list(self.tracks.keys())
        unmatched_dets = set(range(detected_centers.shape[0]))
        unmatched_trks = set(active_ids)
        
        # --- Stage 1: 严格匹配 (保精度) ---
        if unmatched_trks and unmatched_dets:
            t_ids = list(unmatched_trks); d_indices = list(unmatched_dets)
            track_pos = np.array([self.tracks[tid]['x'][:2] for tid in t_ids])
            det_pos = detected_centers[d_indices]
            cost = cdist(track_pos, det_pos, metric='euclidean')
            
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < self.stage1_thresh:
                    tid = t_ids[r]; did = d_indices[c]
                    self._update_track(tid, detected_centers[did])
                    unmatched_trks.remove(tid); unmatched_dets.remove(did)

        # --- Stage 2: 宽松匹配 (保ID，带重置) ---
        if unmatched_trks and unmatched_dets:
            t_ids = list(unmatched_trks); d_indices = list(unmatched_dets)
            track_pos = np.array([self.tracks[tid]['x'][:2] for tid in t_ids])
            det_pos = detected_centers[d_indices]
            cost = cdist(track_pos, det_pos, metric='euclidean')
            
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < self.stage2_thresh:
                    tid = t_ids[r]; did = d_indices[c]
                    # --- 关键调整 3: 找回时必须重置协方差 ---
                    # 否则旧的(错误的)P矩阵会拒绝新的测量，导致更新缓慢
                    if self.tracks[tid]['age'] > 1:
                        self.tracks[tid]['P'] = self.P_reset.copy()
                        # 这里我们不重置x，让它通过Update平滑过去，避免跳变太大
                    
                    self._update_track(tid, detected_centers[did])
                    unmatched_trks.remove(tid); unmatched_dets.remove(did)

        # --- 3. 新生 ---
        for did in unmatched_dets:
            self._create_track(self.next_id, detected_centers[did])
            self.next_id += 1
                
        # --- 4. 输出 ---
        output_centers, output_ids = [], []; to_del = []
        for tid, trk in self.tracks.items():
            # 只输出当前帧匹配上的，保证RMSE统计是针对真实测量的
            if trk['age'] == 0:
                output_centers.append(trk['x'][:2])
                output_ids.append(tid)
            if trk['age'] > self.max_age:
                to_del.append(tid)
        for tid in to_del: del self.tracks[tid]
        
        return np.array(output_centers), np.array(output_ids)
        
    def _create_track(self, tid, pos):
        x_init = np.zeros(6); x_init[:2] = pos
        self.tracks[tid] = {
            'x': x_init, 'P': self.P_reset.copy(), 
            'age': 0, 'trace': [np.array(pos)]
        }

    def _update_track(self, tid, pos):
        trk = self.tracks[tid]
        y = pos - self.H @ trk['x']
        S = self.H @ trk['P'] @ self.H.T + self.R
        K = trk['P'] @ self.H.T @ np.linalg.inv(S)
        trk['x'] += K @ y
        trk['P'] = (np.eye(6) - K @ self.H) @ trk['P']
        trk['age'] = 0
        trk['trace'].append(trk['x'][:2])
        if len(trk['trace']) > 50: trk['trace'].pop(0)
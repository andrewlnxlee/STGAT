import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# ===================================================================
# GNN 后处理器 (配合 GNN-Sharpened DBSCAN 的高精版本)
# ===================================================================
class GNNPostProcessor:
    def __init__(self, dist_thresh=None):
        self.tracks = {}
        self.next_id = 1
        
        # --- 策略参数 ---
        # 既然检测很稳定，我们不需要过长的 max_age，正常值即可
        self.max_age = 5          
        self.stage1_thresh = 40.0 
        self.stage2_thresh = 80.0
        
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
        
        # --- 噪声参数 (回归标准) ---
        # R=5.0: 因为输入是 GNN 锐化后的质心，非常准，所以我们高度信任测量值
        self.R = np.eye(2) * 5.0 
        
        # Q: 正常的机动性允许
        q_pos=0.05; q_vel=0.1; q_acc=0.5
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_acc, q_acc])

        # 初始化协方差
        p_pos=10.0; p_vel=100.0; p_acc=100.0
        self.P_init = np.diag([p_pos, p_pos, p_vel, p_vel, p_acc, p_acc])

    def update(self, detected_centers):
        # 1. 预测
        for trk in self.tracks.values():
            trk['x'] = self.F @ trk['x']
            trk['P'] = self.F @ trk['P'] @ self.F.T + self.Q
            trk['age'] += 1 

        active_ids = list(self.tracks.keys())
        unmatched_dets = set(range(detected_centers.shape[0]))
        unmatched_trks = set(active_ids)
        
        # 2. Stage 1 关联
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

        # 3. Stage 2 关联 (找回)
        if unmatched_trks and unmatched_dets:
            t_ids = list(unmatched_trks); d_indices = list(unmatched_dets)
            track_pos = np.array([self.tracks[tid]['x'][:2] for tid in t_ids])
            det_pos = detected_centers[d_indices]
            cost = cdist(track_pos, det_pos, metric='euclidean')
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < self.stage2_thresh:
                    tid = t_ids[r]; did = d_indices[c]
                    # 简单重置，因为检测很准，不需要太复杂的逻辑
                    if self.tracks[tid]['age'] > 1: self.tracks[tid]['P'] = self.P_init.copy()
                    self._update_track(tid, detected_centers[did])
                    unmatched_trks.remove(tid); unmatched_dets.remove(did)

        # 4. 新生
        for did in unmatched_dets:
            self._create_track(self.next_id, detected_centers[did])
            self.next_id += 1
                
        # 5. 输出
        output_centers, output_ids = [], []; to_del = []
        for tid, trk in self.tracks.items():
            if trk['age'] == 0:
                output_centers.append(trk['x'][:2])
                output_ids.append(tid)
            if trk['age'] > self.max_age:
                to_del.append(tid)
        for tid in to_del: del self.tracks[tid]
        
        return np.array(output_centers), np.array(output_ids)
        
    def _create_track(self, tid, pos):
        x_init = np.zeros(6); x_init[:2] = pos
        self.tracks[tid] = {'x': x_init, 'P': self.P_init.copy(), 'age': 0, 'trace': [np.array(pos)]}

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
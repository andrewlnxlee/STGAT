import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# ===================================================================
# GNN 后处理器 ("Smooth Association, Sharp Output" Strategy)
# ===================================================================
class GNNPostProcessor:
    def __init__(self, dist_thresh=None):
        self.tracks = {}
        self.next_id = 1
        
        # --- 策略参数 ---
        self.max_age = 15          # 保持耐心
        self.stage1_thresh = 40.0 
        self.stage2_thresh = 90.0
        
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
        
        # --- 核心设计：分离关联与输出 ---
        
        # 1. 关联用的 R (极高): 
        # 强迫 KF 忽略单帧抖动，保持预测轨迹的绝对平滑。
        # 这确保了下一帧的关联极其稳定 -> ID Switch 低。
        self.R = np.eye(2) * 100.0 
        
        # 2. 过程噪声 Q (较低): 
        # 假设目标运动是惯性的，减少随机游走。
        q_pos=0.05; q_vel=0.05; q_acc=0.1
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_acc, q_acc])

        # 初始化协方差
        p_pos=20.0; p_vel=200.0; p_acc=600.0
        self.P_init = np.diag([p_pos, p_pos, p_vel, p_vel, p_acc, p_acc])

    def update(self, detected_centers):
        # 1. 预测
        for trk in self.tracks.values():
            # 强力衰减 (防止丢失时预测飞出)
            if trk['age'] > 0:
                trk['x'][4:] *= 0.5 
                trk['x'][2:4] *= 0.9
            
            trk['x'] = self.F @ trk['x']
            trk['P'] = self.F @ trk['P'] @ self.F.T + self.Q
            trk['age'] += 1 

        active_ids = list(self.tracks.keys())
        unmatched_dets = set(range(detected_centers.shape[0]))
        unmatched_trks = set(active_ids)
        
        # 2. 双层关联 (保持不变)
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

        if unmatched_trks and unmatched_dets:
            t_ids = list(unmatched_trks); d_indices = list(unmatched_dets)
            track_pos = np.array([self.tracks[tid]['x'][:2] for tid in t_ids])
            det_pos = detected_centers[d_indices]
            cost = cdist(track_pos, det_pos, metric='euclidean')
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < self.stage2_thresh:
                    tid = t_ids[r]; did = d_indices[c]
                    if self.tracks[tid]['age'] > 1: self.tracks[tid]['P'] = self.P_init.copy()
                    self._update_track(tid, detected_centers[did])
                    unmatched_trks.remove(tid); unmatched_dets.remove(did)

        # 3. 新生
        for did in unmatched_dets:
            self._create_track(self.next_id, detected_centers[did])
            self.next_id += 1
                
        # 4. 输出 (关键修改！)
        output_centers, output_ids = [], []; to_del = []
        for tid, trk in self.tracks.items():
            # 策略：只输出当前帧被 GNN 匹配到的目标
            if trk['age'] == 0:
                # --- 关键修改在这里 ---
                # 我们不输出 KF 的状态 (trk['x'])，因为 R 很大，KF 状态滞后严重。
                # 我们直接输出存储在 trk 里的 'last_meas' (即 GNN 的检测值)。
                # 这样既享受了 KF 关联的稳定性，又享受了 GNN 检测的高精度。
                output_centers.append(trk['last_meas']) 
                output_ids.append(tid)

            if trk['age'] > self.max_age:
                to_del.append(tid)
        for tid in to_del: del self.tracks[tid]
        
        return np.array(output_centers), np.array(output_ids)
        
    def _create_track(self, tid, pos):
        x_init = np.zeros(6); x_init[:2] = pos
        self.tracks[tid] = {
            'x': x_init, 'P': self.P_init.copy(), 
            'age': 0, 
            'trace': [np.array(pos)],
            'last_meas': np.array(pos) # 存储最新测量值
        }

    def _update_track(self, tid, pos):
        trk = self.tracks[tid]
        # 更新 KF 状态 (用于下一帧的关联，非常平滑)
        y = pos - self.H @ trk['x']
        S = self.H @ trk['P'] @ self.H.T + self.R
        K = trk['P'] @ self.H.T @ np.linalg.inv(S)
        trk['x'] += K @ y
        trk['P'] = (np.eye(6) - K @ self.H) @ trk['P']
        
        # 更新记录
        trk['age'] = 0
        trk['last_meas'] = np.array(pos) # 保存高精度测量值用于输出
        
        trk['trace'].append(pos) # 轨迹也记录真实的测量值
        if len(trk['trace']) > 50: trk['trace'].pop(0)
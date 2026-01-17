import numpy as np
from scipy.optimize import linear_sum_assignment

class GNNPostProcessor:
    def __init__(self, dist_thresh=5.0): # Mahalanobis distance
        self.tracks = {}
        self.next_id = 1
        self.max_age = 5
        self.dist_thresh = dist_thresh

    def _mahalanobis_distance(self, x, P, z):
        H = np.array([[1,0,0,0],[0,1,0,0]])
        # FIX 1: 大幅增加测量噪声 R (10.0 -> 50.0)
        # 含义：我们知道 GNN 检测到的质心虽然准，但是会随群形状变化而高频抖动。
        # KF 应该更多地相信自己的预测（平滑轨迹），而不是每一帧都死跟测量值。
        R = np.eye(2) * 50.0 
        y = z - H @ x
        S = H @ P @ H.T + R
        try: return np.sqrt(y.T @ np.linalg.inv(S) @ y)
        except: return 100.0

    def update(self, detected_centers):
        for trk in self.tracks.values():
            F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
            # FIX 2: 保持一定的过程噪声 Q，允许转弯
            # x, y 噪声小一点，vx, vy 噪声大一点，允许速度变化
            Q = np.diag([0.1, 0.1, 2.0, 2.0]) 
            trk['x'] = F @ trk['x']
            trk['P'] = F @ trk['P'] @ F.T + Q
            trk['age'] += 1

        active_ids = list(self.tracks.keys())
        if not active_ids or not detected_centers:
            assignment = {}
            for i, center in enumerate(detected_centers):
                self._create_track(self.next_id, center)
                assignment[i] = self.next_id
                self.next_id += 1
            return assignment

        cost = np.array([[self._mahalanobis_distance(self.tracks[tid]['x'], self.tracks[tid]['P'], det) 
                          for det in detected_centers] for tid in active_ids])
        
        row, col = linear_sum_assignment(cost)
        
        assignment = {}
        used_dets = set()
        
        for r_i, c_i in zip(row, col):
            if cost[r_i, c_i] < self.dist_thresh:
                tid = active_ids[r_i]
                self._update_track(tid, detected_centers[c_i])
                assignment[c_i] = tid
                used_dets.add(c_i)
        
        for i in range(len(detected_centers)):
            if i not in used_dets:
                self._create_track(self.next_id, detected_centers[i])
                assignment[i] = self.next_id
                self.next_id += 1
                
        to_del = [tid for tid in self.tracks if self.tracks[tid]['age'] > self.max_age]
        for tid in to_del: del self.tracks[tid]
            
        return assignment
        
    def _create_track(self, tid, pos):
        # 初始协方差较大，因为刚开始不知道速度
        self.tracks[tid] = {'x': np.array([pos[0], pos[1], 0, 0]), 'P': np.eye(4) * 100, 'age': 0, 'trace': [np.array(pos)]}
        
    def _update_track(self, tid, pos):
        trk = self.tracks[tid]
        H = np.array([[1,0,0,0],[0,1,0,0]])
        # FIX 3: 更新时使用同样的 R
        R = np.eye(2) * 50.0
        
        y = pos - H @ trk['x']
        S = H @ trk['P'] @ H.T + R
        K = trk['P'] @ H.T @ np.linalg.inv(S)
        
        trk['x'] = trk['x'] + K @ y
        trk['P'] = (np.eye(4) - K @ H) @ trk['P']
        trk['age'] = 0
        trk['trace'].append(trk['x'][:2])
        if len(trk['trace']) > 50: trk['trace'].pop(0)
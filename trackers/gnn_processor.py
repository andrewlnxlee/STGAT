import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class GNNPostProcessor:
    def __init__(self, dist_thresh=None):
        self.tracks = {}
        self.next_id = 1
        
        # --- 策略参数 (冠军配置) ---
        self.max_age = 15          
        self.stage1_thresh = 40.0 
        self.stage2_thresh = 90.0
        
        # --- 6D CA (恒定加速度) ---
        dt = 1.0
        self.F = np.array([[1,0,dt,0,0.5*dt**2,0],[0,1,0,dt,0,0.5*dt**2],[0,0,1,0,dt,0],[0,0,0,1,0,dt],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        self.H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])
        
        # --- 核心：高 R 平滑 ---
        self.R = np.eye(2) * 100.0 
        self.Q = np.diag([0.05, 0.05, 0.05, 0.05, 0.1, 0.1])
        self.P_init = np.diag([20.0, 20.0, 200.0, 200.0, 600.0, 600.0])

    def update(self, detected_centers, detected_shapes=None):
        # 1. 预测
        for trk in self.tracks.values():
            if trk['age'] > 0: 
                trk['x'][4:] *= 0.5; trk['x'][2:4] *= 0.9 # 衰减
            trk['x'] = self.F @ trk['x']
            trk['P'] = self.F @ trk['P'] @ self.F.T + self.Q
            trk['age'] += 1

        active_ids = list(self.tracks.keys())
        unmatched_dets = set(range(detected_centers.shape[0]))
        unmatched_trks = set(active_ids)
        
        # 2. 关联 (纯位置，不看形状)
        # 形状信息只用于透传输出，不参与 Cost 计算，防止干扰
        def associate(thresh, t_set, d_set):
            if not t_set or not d_set: return
            t_ids = list(t_set); d_ids = list(d_set)
            t_pos = np.array([self.tracks[tid]['x'][:2] for tid in t_ids])
            d_pos = detected_centers[d_ids]
            
            cost = cdist(t_pos, d_pos, metric='euclidean')
            row, col = linear_sum_assignment(cost)
            
            for r, c in zip(row, col):
                if cost[r, c] < thresh:
                    tid = t_ids[r]; did = d_ids[c]
                    if self.tracks[tid]['age'] > 1: self.tracks[tid]['P'] = self.P_init.copy()
                    
                    # 记录形状
                    sh = detected_shapes[did] if detected_shapes is not None else None
                    self._update_track(tid, detected_centers[did], sh)
                    
                    if tid in unmatched_trks: unmatched_trks.remove(tid)
                    if did in unmatched_dets: unmatched_dets.remove(did)

        # Stage 1 & 2
        associate(self.stage1_thresh, unmatched_trks, unmatched_dets)
        associate(self.stage2_thresh, unmatched_trks, unmatched_dets)

        # 3. 新生
        for did in unmatched_dets:
            sh = detected_shapes[did] if detected_shapes is not None else None
            self._create_track(self.next_id, detected_centers[did], sh)
            self.next_id += 1
                
        # 4. 输出
        out_c, out_id, out_sh = [], [], []
        to_del = []
        for tid, trk in self.tracks.items():
            if trk['age'] == 0:
                out_c.append(trk['last_meas'])
                out_id.append(tid)
                out_sh.append(trk['shape'])
            if trk['age'] > self.max_age: to_del.append(tid)
        for tid in to_del: del self.tracks[tid]
        
        return np.array(out_c), np.array(out_id), np.array(out_sh)
        
    def _create_track(self, tid, pos, shape):
        init_shape = shape if shape is not None else np.array([3.0, 3.0])
        self.tracks[tid] = {'x': np.zeros(6), 'P': self.P_init.copy(), 'age': 0, 'trace': [pos], 'last_meas': pos, 'shape': init_shape}
        self.tracks[tid]['x'][:2] = pos

    def _update_track(self, tid, pos, shape):
        trk = self.tracks[tid]
        y = pos - self.H @ trk['x']; S = self.H @ trk['P'] @ self.H.T + self.R
        K = trk['P'] @ self.H.T @ np.linalg.inv(S)
        trk['x'] += K @ y; trk['P'] = (np.eye(6) - K @ self.H) @ trk['P']
        trk['age'] = 0; trk['last_meas'] = pos
        if shape is not None: trk['shape'] = 0.9 * trk['shape'] + 0.1 * shape
        trk['trace'].append(pos)
        if len(trk['trace']) > 50: trk['trace'].pop(0)
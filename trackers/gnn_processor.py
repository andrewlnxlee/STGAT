import numpy as np
from scipy.optimize import linear_sum_assignment

class GNNPostProcessor:
    def __init__(self, dist_thresh=5.0):
        self.tracks = {}
        self.next_id = 1
        self.max_age = 5
        self.dist_thresh = dist_thresh

    def _mahalanobis_distance(self, x, P, z):
        H = np.array([[1,0,0,0],[0,1,0,0]])
        R = np.eye(2) * 10.0 # Measurement Noise
        y = z - H @ x
        S = H @ P @ H.T + R
        try:
            return np.sqrt(y.T @ np.linalg.inv(S) @ y)
        except:
            return 100.0

    def update(self, detected_centers):
        # 1. Predict
        for trk in self.tracks.values():
            F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
            Q = np.diag([0.5, 0.5, 2.0, 2.0]) # Process Noise
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

        # 2. Associate (Mahalanobis)
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
        
        # 3. Create New
        for i in range(len(detected_centers)):
            if i not in used_dets:
                self._create_track(self.next_id, detected_centers[i])
                assignment[i] = self.next_id
                self.next_id += 1
        
        # 4. Clean up
        to_del = [tid for tid in self.tracks if self.tracks[tid]['age'] > self.max_age]
        for tid in to_del: del self.tracks[tid]
            
        return assignment
        
    def _create_track(self, tid, pos):
        self.tracks[tid] = {'x': np.array([pos[0], pos[1], 0, 0]), 'P': np.eye(4) * 50, 'age': 0, 'trace': [np.array(pos)]}
        
    def _update_track(self, tid, pos):
        trk = self.tracks[tid]
        H = np.array([[1,0,0,0],[0,1,0,0]])
        R = np.eye(2) * 10.0
        y = pos - H @ trk['x']
        S = H @ trk['P'] @ H.T + R
        K = trk['P'] @ H.T @ np.linalg.inv(S)
        trk['x'] = trk['x'] + K @ y
        trk['P'] = (np.eye(4) - K @ H) @ trk['P']
        trk['age'] = 0
        trk['trace'].append(trk['x'][:2])
        if len(trk['trace']) > 50: trk['trace'].pop(0)
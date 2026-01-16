import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from trackers.kalman_box import KalmanBoxTracker

class BaselineTracker:
    def __init__(self, eps=35, min_samples=3):
        self.cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
        self.reset()

    def reset(self):
        self.trackers = []
        KalmanBoxTracker.count = 0
        self.count_frames = 0

    def step(self, points):
        self.count_frames += 1
        for trk in self.trackers:
            trk.predict()
            
        if len(points) == 0:
            ret_c, ret_id = [], []
            for trk in reversed(self.trackers):
                if trk.time_since_update > 5: self.trackers.pop(self.trackers.index(trk))
            return np.array(ret_c), np.array(ret_id), np.array([])

        labels = self.cluster_model.fit_predict(points)
        detections = [np.mean(points[labels == l], axis=0) for l in set(labels) if l != -1]
        
        matched, unmatched_dets, _ = self._associate(self.trackers, detections)
        
        for t, d in matched: self.trackers[t].update(detections[d])
        for i in unmatched_dets: self.trackers.append(KalmanBoxTracker(detections[i]))
        
        ret_c, ret_id = [], []
        for trk in reversed(self.trackers):
            if trk.time_since_update < 1 and (trk.hit_streak >= 2 or self.count_frames <= 2):
                ret_c.append(trk.x[:2])
                ret_id.append(trk.id)
            if trk.time_since_update > 5:
                self.trackers.pop(self.trackers.index(trk))
        return np.array(ret_c), np.array(ret_id), labels

    def _associate(self, trackers, detections):
        if not trackers: return [], list(range(len(detections))), []
        cost = np.array([[np.linalg.norm(trk.x[:2] - det) for det in detections] for trk in trackers])
        r, c = linear_sum_assignment(cost)
        m, ud, ut = [], list(range(len(detections))), list(range(len(trackers)))
        for r_i, c_i in zip(r, c):
            if cost[r_i, c_i] < 50.0:
                m.append((r_i, c_i))
                if c_i in ud: ud.remove(c_i)
                if r_i in ut: ut.remove(r_i)
        return m, ud, ut
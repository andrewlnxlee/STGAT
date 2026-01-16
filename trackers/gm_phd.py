import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances

class GaussianComponent:
    def __init__(self, weight, mean, cov):
        self.w = weight
        self.m = mean
        self.P = cov

class GMPHDTracker:
    def __init__(self):
        self.p_survival = 0.99
        self.p_detect = 0.98
        self.clutter_density = 1e-5
        self.birth_weight = 0.05
        self.prune_thresh = 1e-4
        self.extract_thresh = 0.5
        self.max_gaussians = 100
        self.next_id = 1
        self.active_tracks = {}
        self.components = []

    def reset(self):
        self.components = []
        self.next_id = 1
        self.active_tracks = {}

    def step(self, measurements):
        F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        Q = np.eye(4) * 2.0
        
        # 1. Predict
        predicted_components = []
        for comp in self.components:
            predicted_components.append(GaussianComponent(self.p_survival * comp.w, F @ comp.m, F @ comp.P @ F.T + Q))
        
        for z in measurements:
            m_birth = np.array([z[0], z[1], 0, 0])
            P_birth = np.eye(4) * 50.0
            predicted_components.append(GaussianComponent(self.birth_weight, m_birth, P_birth))
        self.components = predicted_components

        # 2. Update
        updated_components = []
        H = np.array([[1,0,0,0],[0,1,0,0]])
        R = np.eye(2) * 5.0
        
        for comp in self.components:
            updated_components.append(GaussianComponent((1 - self.p_detect) * comp.w, comp.m, comp.P))
            
        for z in measurements:
            z_comps = []; total_likelihood = 0.0
            for comp in self.components:
                y = z - H @ comp.m
                S = H @ comp.P @ H.T + R
                K = comp.P @ H.T @ np.linalg.inv(S)
                try: likelihood = multivariate_normal.pdf(z, mean=H@comp.m, cov=S)
                except: likelihood = 0.0
                z_comps.append(GaussianComponent(self.p_detect * comp.w * likelihood, comp.m + K @ y, (np.eye(4) - K @ H) @ comp.P))
                total_likelihood += z_comps[-1].w
            
            norm = self.clutter_density + total_likelihood
            for z_c in z_comps:
                z_c.w /= norm
                updated_components.append(z_c)
        self.components = updated_components
        
        # 3. Pruning & Merging
        self.components.sort(key=lambda x: x.w, reverse=True)
        self.components = [c for c in self.components if c.w > self.prune_thresh]
        
        merged_components = []
        while len(self.components) > 0:
            high_w = self.components[0]
            close_indices = [0]
            for i in range(1, len(self.components)):
                if np.linalg.norm(high_w.m[:2] - self.components[i].m[:2]) < 30.0:
                    close_indices.append(i)
            
            merged_w = sum(self.components[i].w for i in close_indices)
            merged_m = sum(self.components[i].w * self.components[i].m for i in close_indices) / merged_w
            merged_components.append(GaussianComponent(merged_w, merged_m, high_w.P))
            self.components = [c for i, c in enumerate(self.components) if i not in close_indices]
        
        self.components = merged_components[:self.max_gaussians]
        
        # 4. State Extraction & Labeling
        extracted = [comp.m[:2] for comp in self.components if comp.w > self.extract_thresh]
        ret_c, ret_id = [], []
        
        if extracted:
            if not self.active_tracks:
                for s in extracted:
                    ret_c.append(s); ret_id.append(self.next_id)
                    self.active_tracks[self.next_id] = s; self.next_id += 1
            else:
                prev_ids = list(self.active_tracks.keys())
                prev_pos = list(self.active_tracks.values())
                cost = euclidean_distances(prev_pos, extracted)
                row, col = linear_sum_assignment(cost)
                
                assigned_idx = set()
                new_tracks = {}
                for r, c in zip(row, col):
                    if cost[r, c] < 60.0:
                        tid = prev_ids[r]
                        pos = extracted[c]
                        ret_c.append(pos); ret_id.append(tid)
                        new_tracks[tid] = pos; assigned_idx.add(c)
                
                for i, pos in enumerate(extracted):
                    if i not in assigned_idx:
                        ret_c.append(pos); ret_id.append(self.next_id)
                        new_tracks[self.next_id] = pos; self.next_id += 1
                self.active_tracks = new_tracks
        else:
            self.active_tracks = {}
            
        return np.array(ret_c), np.array(ret_id)
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances

class GaussianComponent:
    def __init__(self, weight, mean, cov):
        self.w = weight
        self.m = mean
        self.P = cov

class GMCPHDTracker:
    """
    Gaussian Mixture Cardinalityized PHD (Simplified Engineering Version)
    Differs from PHD by actively maintaining cardinality statistics to stabilize the weight update.
    """
    def __init__(self):
        # Model Parameters
        self.p_survival = 0.99
        self.p_detect = 0.98
        self.clutter_density = 1e-5
        self.birth_weight = 0.05
        
        # Thresholds
        self.prune_thresh = 1e-4
        self.merge_thresh = 30.0
        self.max_gaussians = 100
        self.extract_thresh = 0.5
        
        # Cardinality Distribution (Mean and Variance of target number)
        self.N_mean = 0.0
        self.N_var = 0.0
        
        # ID Management
        self.next_id = 1
        self.active_tracks = {} 
        self.components = []

    def reset(self):
        self.components = []
        self.next_id = 1
        self.active_tracks = {}
        self.N_mean = 0.0
        self.N_var = 0.0

    def step(self, measurements):
        # --- 1. Prediction ---
        F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        Q = np.eye(4) * 2.0
        
        predicted_comps = []
        # Survival
        w_survive_sum = 0
        for comp in self.components:
            m_pred = F @ comp.m
            P_pred = F @ comp.P @ F.T + Q
            w_pred = self.p_survival * comp.w
            predicted_comps.append(GaussianComponent(w_pred, m_pred, P_pred))
            w_survive_sum += w_pred
            
        # Birth (Adaptive at measurements)
        w_birth_sum = 0
        H = np.array([[1,0,0,0],[0,1,0,0]])
        for z in measurements:
            m_birth = np.array([z[0], z[1], 0, 0])
            P_birth = np.eye(4) * 50.0
            # Weight initialization is crucial
            predicted_comps.append(GaussianComponent(self.birth_weight, m_birth, P_birth))
            w_birth_sum += self.birth_weight
            
        # Cardinality Prediction (Simplified)
        # N_k|k-1 = P_s * N_k-1 + N_birth
        self.N_mean = self.p_survival * self.N_mean + w_birth_sum
        self.N_var = self.p_survival**2 * self.N_var + w_birth_sum # Approximate
        
        self.components = predicted_comps

        # --- 2. Update (CPHD Scaling) ---
        updated_comps = []
        
        # Pre-calculate likelihoods and clutter
        R = np.eye(2) * 5.0
        
        # Miss detection term (Corrected by CPHD factor)
        # In full CPHD, this scales based on cardinality. 
        # Here we use a simpler heuristic: if we think N is high but sum(w) is low, boost weights.
        current_w_sum = sum(c.w for c in self.components)
        cphd_factor = 1.0 
        if current_w_sum > 0:
            cphd_factor = self.N_mean / (current_w_sum + 1e-6)
            
        for comp in self.components:
            w_miss = (1 - self.p_detect) * comp.w * cphd_factor # Apply correction
            updated_comps.append(GaussianComponent(w_miss, comp.m, comp.P))
            
        # Detection term
        for z in measurements:
            z_comps = []
            total_likelihood = 0.0
            
            for comp in self.components:
                # Kalman Innovation
                y = z - H @ comp.m
                S = H @ comp.P @ H.T + R
                K = comp.P @ H.T @ np.linalg.inv(S)
                m_upd = comp.m + K @ y
                P_upd = (np.eye(4) - K @ H) @ comp.P
                
                try: likelihood = multivariate_normal.pdf(z, mean=H@comp.m, cov=S)
                except: likelihood = 0.0
                
                w_upd = self.p_detect * comp.w * likelihood
                z_comps.append(GaussianComponent(w_upd, m_upd, P_upd))
                total_likelihood += w_upd
            
            # CPHD Normalization: Balances clutter vs detection
            # Full CPHD uses symmetric functions here. 
            # We approximate: Normalized by (Clutter + Likelihood) but scaled by cardinality belief.
            norm = self.clutter_density + total_likelihood
            
            for z_c in z_comps:
                z_c.w /= norm
                updated_comps.append(z_c)
                
        self.components = updated_comps
        
        # Update Cardinality Estimate
        total_w = sum(c.w for c in self.components)
        self.N_mean = total_w # In GM-PHD, N = sum(w)
        self.N_var = total_w  # Poisson assumption
        
        # --- 3. Pruning & Merging ---
        self.components.sort(key=lambda x: x.w, reverse=True)
        self.components = [c for c in self.components if c.w > self.prune_thresh]
        
        merged_components = []
        while len(self.components) > 0:
            high_w_comp = self.components[0]
            close_indices = [0]
            for i in range(1, len(self.components)):
                if np.linalg.norm(high_w_comp.m[:2] - self.components[i].m[:2]) < self.merge_thresh:
                    close_indices.append(i)
            
            merged_w = sum(self.components[i].w for i in close_indices)
            merged_m = sum(self.components[i].w * self.components[i].m for i in close_indices) / merged_w
            merged_components.append(GaussianComponent(merged_w, merged_m, high_w_comp.P))
            self.components = [c for i, c in enumerate(self.components) if i not in close_indices]
            
        self.components = merged_components[:self.max_gaussians]
        
        # --- 4. Extraction & Labeling ---
        extracted = [comp.m[:2] for comp in self.components if comp.w > self.extract_thresh]
        return self._assign_labels(extracted)

    def _assign_labels(self, extracted_pos):
        ret_c, ret_id = [], []
        if not extracted_pos:
            self.active_tracks = {}
            return np.array(ret_c), np.array(ret_id)
            
        if not self.active_tracks:
            for pos in extracted_pos:
                ret_c.append(pos); ret_id.append(self.next_id)
                self.active_tracks[self.next_id] = pos
                self.next_id += 1
        else:
            prev_ids = list(self.active_tracks.keys())
            prev_pos = list(self.active_tracks.values())
            
            cost = euclidean_distances(prev_pos, extracted_pos)
            row, col = linear_sum_assignment(cost)
            
            assigned_idx = set()
            new_tracks = {}
            
            for r, c in zip(row, col):
                if cost[r, c] < 50.0:
                    tid = prev_ids[r]
                    pos = extracted_pos[c]
                    ret_c.append(pos); ret_id.append(tid)
                    new_tracks[tid] = pos
                    assigned_idx.add(c)
                    
            for i, pos in enumerate(extracted_pos):
                if i not in assigned_idx:
                    ret_c.append(pos); ret_id.append(self.next_id)
                    new_tracks[self.next_id] = pos
                    self.next_id += 1
            self.active_tracks = new_tracks
            
        return np.array(ret_c), np.array(ret_id)
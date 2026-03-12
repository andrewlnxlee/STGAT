import os
import torch
import numpy as np
import pickle
import sys
import networkx as nx
from scipy.optimize import linear_sum_assignment

import importlib.util

# 使用 importlib 手动加载，避免与根目录的 model.py 冲突
_model_path = os.path.join(os.getcwd(), 'Social-STGCNN-master', 'model.py')
_spec = importlib.util.spec_from_file_location("social_stgcnn_internal_model", _model_path)
_social_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_social_mod)
social_stgcnn = _social_mod.social_stgcnn

class SocialSTGCNNTracker:
    """
    Social-STGCNN 跟踪包装器
    - 加载预训练的 ST-GCN 模型
    - 每一帧实时构建社会交互图
    - 使用预测的未来位置进行数据关联
    """
    def __init__(self, scene='eth', max_distance=50.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_distance = max_distance
        self.obs_len = 8
        self.tracks = {}
        self.next_id = 1
        
        # 自动定位场景对应的 checkpoint
        ckpt_dir = os.path.join('Social-STGCNN-master', 'checkpoint', f'social-stgcnn-{scene}')
        model_path = os.path.join(ckpt_dir, 'val_best.pth')
        args_path = os.path.join(ckpt_dir, 'args.pkl')
        
        if not os.path.exists(model_path):
            print(f"⚠️ [Social-STGCNN] 未找到场景 {scene} 的权重: {model_path}")
            self.model = None
            return

        with open(args_path, 'rb') as f:
            args = pickle.load(f)
            
        self.model = social_stgcnn(
            n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
            output_feat=args.output_size, seq_len=args.obs_seq_len,
            kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        # print(f"✅ [Social-STGCNN] 已加载 {scene} 场景预训练模型")

    def _anorm(self, p1, p2):
        norm = np.linalg.norm(p1 - p2)
        return 0 if norm == 0 else 1.0 / norm

    def _build_graph(self, active_tids):
        """为当前所有活跃轨迹构建 Social-STGCNN 所需的 V 和 A 矩阵"""
        n_nodes = len(active_tids)
        # Social-STGCNN 需要相对坐标 (dx, dy)
        v = np.zeros((self.obs_len, n_nodes, 2))
        a = np.zeros((self.obs_len, n_nodes, n_nodes))
        
        for i, tid in enumerate(active_tids):
            hist = self.tracks[tid]['history']
            # 补齐长度到 obs_len
            if len(hist) < self.obs_len:
                hist = [hist[0]] * (self.obs_len - len(hist)) + hist
            else:
                hist = hist[-self.obs_len:]
            
            # 转换为相对坐标
            hist_arr = np.array(hist)
            rel_hist = np.zeros_like(hist_arr)
            rel_hist[1:] = hist_arr[1:] - hist_arr[:-1]
            v[:, i, :] = rel_hist

        # 构建每一帧的归一化拉普拉斯邻接矩阵
        for t in range(self.obs_len):
            step_rel = v[t]
            adj = np.zeros((n_nodes, n_nodes))
            for h in range(n_nodes):
                adj[h, h] = 1
                for k in range(h + 1, n_nodes):
                    val = self._anorm(step_rel[h], step_rel[k])
                    adj[h, k] = val
                    adj[k, h] = val
            
            # 归一化
            if n_nodes > 0:
                G = nx.from_numpy_matrix(adj)
                a[t, :, :] = nx.normalized_laplacian_matrix(G).toarray()
        
        return (torch.from_numpy(v).float().to(self.device), 
                torch.from_numpy(a).float().to(self.device))

    def step(self, measurements):
        if self.model is None or len(measurements) == 0:
            measurements = np.empty((0, 2)) if len(measurements) == 0 else measurements
            
        active_tids = list(self.tracks.keys())
        predictions = {}

        # --- 定义卡尔曼滤波参数 ---
        F = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        Q = np.diag([0.1, 0.1, 1.0, 1.0])
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        R = np.eye(2) * 15.0 # 设置适度的观测噪声，保证平滑性

        # 1. 预测
        if len(active_tids) > 0 and self.model is not None:
            v, a = self._build_graph(active_tids)
            # Input format: (Batch, Feat, Seq, Node)
            v_in = v.permute(2, 0, 1).unsqueeze(0) # (1, 2, 8, N)
            with torch.no_grad():
                v_pred, _ = self.model(v_in, a) # (1, 5, 12, N)
                # v_pred: [0:2] 是均值 dx, dy
                v_pred = v_pred.squeeze(0).permute(1, 2, 0).cpu().numpy() # (12, N, 5)
            
            for i, tid in enumerate(active_tids):
                last_pos = self.tracks[tid]['history'][-1]
                pred_dx_dy = v_pred[0, i, :2] # 取未来第一步预测
                # STGCNN的预测用于匈牙利关联
                predictions[tid] = last_pos + pred_dx_dy
                
                # KF预测步
                self.tracks[tid]['kf_x'] = F @ self.tracks[tid]['kf_x']
                self.tracks[tid]['kf_P'] = F @ self.tracks[tid]['kf_P'] @ F.T + Q
        else:
            for tid in active_tids:
                predictions[tid] = self.tracks[tid]['history'][-1]
                self.tracks[tid]['kf_x'] = F @ self.tracks[tid]['kf_x']
                self.tracks[tid]['kf_P'] = F @ self.tracks[tid]['kf_P'] @ F.T + Q

        # 2. 关联 (匈牙利算法)
        n_meas = len(measurements)
        n_tracks = len(active_tids)
        cost_matrix = np.full((n_tracks, n_meas), 1000.0)
        
        for r, tid in enumerate(active_tids):
            pred_pos = predictions[tid]
            for c in range(n_meas):
                dist = np.linalg.norm(pred_pos - measurements[c])
                if dist < self.max_distance:
                    cost_matrix[r, c] = dist
                    
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_tracks = set()
        matched_meas = set()
        point_labels = np.full(n_meas, -1, dtype=int)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.max_distance:
                tid = active_tids[r]
                trk = self.tracks[tid]
                
                # KF更新步
                z = measurements[c]
                y = z - H @ trk['kf_x']
                S = H @ trk['kf_P'] @ H.T + R
                K = trk['kf_P'] @ H.T @ np.linalg.inv(S)
                trk['kf_x'] = trk['kf_x'] + K @ y
                trk['kf_P'] = (np.eye(4) - K @ H) @ trk['kf_P']

                # 将KF平滑后的坐标送入STGCNN作为历史
                trk['history'].append(trk['kf_x'][:2].copy())
                trk['misses'] = 0
                trk['hit_streak'] += 1
                matched_tracks.add(tid)
                matched_meas.add(c)
                point_labels[c] = tid 

        # 3. 新生和死亡管理
        for c in range(n_meas):
            if c not in matched_meas:
                new_id = self.next_id
                self.next_id += 1
                kf_x = np.array([measurements[c][0], measurements[c][1], 0, 0], dtype=float)
                kf_P = np.eye(4) * 50.0
                self.tracks[new_id] = {
                    'history': [measurements[c]], 
                    'misses': 0, 
                    'hit_streak': 1,
                    'kf_x': kf_x,
                    'kf_P': kf_P
                }
                point_labels[c] = new_id

        to_del = []
        for tid in active_tids:
            if tid not in matched_tracks:
                self.tracks[tid]['misses'] += 1
                if self.tracks[tid]['misses'] > 3:
                    to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]
            
        out_centers, out_ids = [], []
        for c, meas in enumerate(measurements):
            tid = point_labels[c]
            # 引入KF后，直接输出KF平滑的状态
            if tid in self.tracks and self.tracks[tid]['hit_streak'] >= 2:
                out_centers.append(self.tracks[tid]['kf_x'][:2])
                out_ids.append(tid)
                
        return np.array(out_centers), np.array(out_ids), point_labels

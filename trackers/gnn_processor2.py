# trackers/gnn_processor.py
# 升级版 V2: 点级跟踪 + GNN特征嵌入关联 + Temporal Attention 运动预测
import torch
import torch.nn as nn
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# ---------------------------------------------------------
# 轻量级 Temporal Attention 运动预测器
# 与 SOTA-Transformer 同架构，但更紧密集成到 GNN 管线中
# ---------------------------------------------------------
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class _MotionPredictor(nn.Module):
    """基于 Multi-Head Attention 的运动预测器，利用速度历史预测下一步位移"""
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        x_emb = self.input_proj(x)
        x_emb = self.pos_encoder(x_emb)
        out = self.transformer_encoder(x_emb)
        return self.output_proj(out[:, -1, :])


class GNNPostProcessor:
    """
    V2: 点级跟踪器
    - 直接在原始/校正后测量点上做匹配
    - 利用 GNN 的高维特征嵌入计算关联代价
    - 内嵌 Temporal Attention 运动预测器
    """
    def __init__(self, dist_thresh=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 内嵌运动预测器（与 SOTA-Transformer 同架构）
        self.seq_length = 4
        self.motion_predictor = _MotionPredictor().to(self.device)
        self._auto_train_motion()
        self.motion_predictor.eval()

        self.tracks = {}
        self.next_id = 1

        # 关联参数
        self.max_distance = 45.0      # 稍微收紧关联门限
        self.max_distance_stage2 = 70.0
        self.max_age = 5              # 增加容忍度

        # 调优后的关联权重
        self.w_dist = 1.0             
        self.w_feat = 10.0             
        
        # 纠偏平滑因子 (移除人工干预，完全信任 GNN 的纠偏能力)
        self.offset_alpha = 1.0       

    def _auto_train_motion(self, epochs=400):
        """预训练运动预测器（同 SOTA-Transformer 的训练方式）"""
        print(f"\n[H-GAT-GT V2] 正在训练内嵌 Temporal Attention 运动预测器...")
        optimizer = torch.optim.AdamW(self.motion_predictor.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.MSELoss()
        self.motion_predictor.train()

        batch_size = 512
        for epoch in range(epochs):
            accel = torch.randn(batch_size, 2).to(self.device) * 0.5
            v0 = torch.randn(batch_size, 2).to(self.device) * 5.0

            v_seq = []
            v_curr = v0
            for _ in range(self.seq_length + 1):
                v_seq.append(v_curr)
                v_curr = v_curr + accel + torch.randn(batch_size, 2).to(self.device) * 0.1

            v_seq = torch.stack(v_seq, dim=1)
            x_train = v_seq[:, :-1, :]
            y_train = v_seq[:, -1, :]

            optimizer.zero_grad()
            pred_v = self.motion_predictor(x_train)
            loss = criterion(pred_v, y_train)
            loss.backward()
            optimizer.step()

        print(f"[H-GAT-GT V2] 运动预测器训练完成! Final MSE Loss: {loss.item():.4f}\n")

    def _predict_position(self, track):
        """用 Temporal Attention 预测下一帧位置"""
        hist = track['history']
        if len(hist) < 2:
            return hist[-1]

        vels = []
        for i in range(1, len(hist)):
            vels.append(hist[i] - hist[i - 1])

        if len(vels) > self.seq_length:
            vels = vels[-self.seq_length:]
        while len(vels) < self.seq_length:
            vels.insert(0, vels[0])

        with torch.no_grad():
            v_tensor = torch.tensor(np.array([vels]), dtype=torch.float32).to(self.device)
            pred_v = self.motion_predictor(v_tensor).cpu().numpy()[0]

        return hist[-1] + pred_v

    def update(self, detected_points, point_features=None):
        """
        点级跟踪更新
        Args:
            detected_points: [N, 2] 校正后的测量点坐标
            point_features: [N, D] GNN 输出的节点特征嵌入 (可选)
        Returns:
            out_centers: [M, 2] 已确认的跟踪点坐标
            out_ids: [M] 对应的跟踪 ID
            point_labels: [N] 每个输入点被分配到的跟踪 ID (-1 = 未分配)
        """
        if len(detected_points) == 0:
            detected_points = np.empty((0, 2))

        # --- 步骤 1: 用 Temporal Attention 预测现有轨迹的未来位置 ---
        predictions = {}
        for tid, track in self.tracks.items():
            predictions[tid] = self._predict_position(track)

        # --- 步骤 2: 构建关联代价矩阵 ---
        active_tids = list(self.tracks.keys())
        n_tracks = len(active_tids)
        n_dets = len(detected_points)

        point_labels = np.full(n_dets, -1, dtype=int)

        if n_tracks > 0 and n_dets > 0:
            # 距离代价
            pred_positions = np.array([predictions[tid] for tid in active_tids])
            dist_cost = cdist(pred_positions, detected_points, metric='euclidean')

            # 特征余弦距离代价
            feat_cost = np.zeros_like(dist_cost)
            if point_features is not None:
                for r, tid in enumerate(active_tids):
                    if 'feature' in self.tracks[tid] and self.tracks[tid]['feature'] is not None:
                        trk_feat = self.tracks[tid]['feature']  # [D]
                        # 余弦距离 = 1 - 余弦相似度
                        for c in range(n_dets):
                            det_feat = point_features[c]
                            cos_sim = np.dot(trk_feat, det_feat) / (
                                np.linalg.norm(trk_feat) * np.linalg.norm(det_feat) + 1e-8
                            )
                            feat_cost[r, c] = 1.0 - cos_sim  # [0, 2]

            total_cost = self.w_dist * dist_cost + self.w_feat * feat_cost

            # --- 两阶段关联（先严格后宽松）---
            matched_tracks = set()
            matched_dets = set()

            # Stage 1: 严格门限
            row_ind, col_ind = linear_sum_assignment(total_cost)
            for r, c in zip(row_ind, col_ind):
                if dist_cost[r, c] < self.max_distance:
                    tid = active_tids[r]
                    self._update_track(tid, detected_points[c],
                                       point_features[c] if point_features is not None else None)
                    matched_tracks.add(tid)
                    matched_dets.add(c)
                    point_labels[c] = tid

            # Stage 2: 宽松门限处理剩余
            unmatched_trk_ids = [tid for tid in active_tids if tid not in matched_tracks]
            unmatched_det_ids = [c for c in range(n_dets) if c not in matched_dets]

            if len(unmatched_trk_ids) > 0 and len(unmatched_det_ids) > 0:
                sub_pred = np.array([predictions[tid] for tid in unmatched_trk_ids])
                sub_det = detected_points[unmatched_det_ids]
                sub_dist = cdist(sub_pred, sub_det, metric='euclidean')

                sub_row, sub_col = linear_sum_assignment(sub_dist)
                for r2, c2 in zip(sub_row, sub_col):
                    if sub_dist[r2, c2] < self.max_distance_stage2:
                        tid = unmatched_trk_ids[r2]
                        did = unmatched_det_ids[c2]
                        self._update_track(tid, detected_points[did],
                                           point_features[did] if point_features is not None else None)
                        matched_tracks.add(tid)
                        matched_dets.add(did)
                        point_labels[did] = tid

            # --- 步骤 3: 新生轨迹 ---
            for c in range(n_dets):
                if c not in matched_dets:
                    new_id = self.next_id
                    self.next_id += 1
                    feat = point_features[c] if point_features is not None else None
                    self.tracks[new_id] = {
                        'history': [detected_points[c].copy()],
                        'age': 0,
                        'misses': 0,
                        'hit_streak': 1,
                        'feature': feat,
                    }
                    point_labels[c] = new_id

        elif n_dets > 0:
            # 没有活跃轨迹，全部创建新轨迹
            for c in range(n_dets):
                new_id = self.next_id
                self.next_id += 1
                feat = point_features[c] if point_features is not None else None
                self.tracks[new_id] = {
                    'history': [detected_points[c].copy()],
                    'age': 0,
                    'misses': 0,
                    'hit_streak': 1,
                    'feature': feat,
                }
                point_labels[c] = new_id

        # --- 步骤 4: 清理丢失轨迹 ---
        to_del = []
        for tid in active_tids:
            if tid not in (matched_tracks if n_tracks > 0 and n_dets > 0 else set()):
                self.tracks[tid]['misses'] += 1
                self.tracks[tid]['hit_streak'] = 0
                if self.tracks[tid]['misses'] > self.max_age:
                    to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        # --- 步骤 5: 输出已确认的跟踪结果 ---
        out_centers = []
        out_ids = []
        for c in range(n_dets):
            tid = point_labels[c]
            if tid > 0 and tid in self.tracks and self.tracks[tid]['hit_streak'] >= 1:
                # 移除基于预测的时序滤波，因为它在强噪声下可能导致误差累积（发散）
                # 直接输出经过轻度 GNN 纠偏的点
                out_centers.append(detected_points[c])
                out_ids.append(tid)

        return np.array(out_centers).reshape(-1, 2), np.array(out_ids), point_labels

    def _update_track(self, tid, pos, feature=None):
        """更新已有轨迹"""
        trk = self.tracks[tid]
        trk['history'].append(pos.copy())
        if len(trk['history']) > self.seq_length + 2:
            trk['history'].pop(0)
        trk['age'] = 0
        trk['misses'] = 0
        trk['hit_streak'] = trk.get('hit_streak', 0) + 1

        # EMA 更新特征嵌入
        if feature is not None:
            if trk.get('feature') is not None:
                trk['feature'] = 0.7 * trk['feature'] + 0.3 * feature
            else:
                trk['feature'] = feature.copy()
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import math

# ---------------------------------------------------------
# 1. 深度学习框架：基于 Transformer 的序列速度推演
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0) # (1, max_len, d_model)

    def forward(self, x):
        pos_emb = self.pe[:, :x.size(1), :].to(x.device)
        return x + pos_emb

class MotionTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=2):
        super(MotionTransformer, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, input_dim) # 预测 dx, dy
        )

    def forward(self, x):
        # x: (batch_size, seq_length, 2)
        x_emb = self.input_proj(x)
        x_emb = self.pos_encoder(x_emb)
        out = self.transformer_encoder(x_emb)
        # 提取序列最后的特征作为下一个时间步速度预测的核心
        pred = self.output_proj(out[:, -1, :])
        return pred

# ---------------------------------------------------------
# 2. Transformer 神经追踪器主类 (SOTA Baseline)
# ---------------------------------------------------------
class SOTATransformerTracker:
    def __init__(self, seq_length=4, max_distance=50.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_length = seq_length
        self.max_distance = max_distance
        
        # 激活自我学习的 Transformer 注意力网络
        self.model = MotionTransformer().to(self.device)
        self._auto_train()
        self.model.eval()

        self.tracks = {}
        self.next_id = 1

    def _auto_train(self, epochs=400):
        """
        [自学习机制]：追踪器首次启动时，利用海量生成的物理曲线轨迹，
        训练多头注意力机制（Multi-Head Attention）理解加速度和惯性，从而形成直觉。
        """
        print(f"\n[SOTA-Transformer] 正在初始化多头注意力轨迹追踪基线...进行物理预训练...")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.MSELoss()
        self.model.train()
        
        batch_size = 512
        for epoch in range(epochs):
            accel = torch.randn(batch_size, 2).to(self.device) * 0.5
            v0 = torch.randn(batch_size, 2).to(self.device) * 5.0
            
            v_seq = []
            v_curr = v0
            for _ in range(self.seq_length + 1):
                v_seq.append(v_curr)
                v_curr = v_curr + accel + torch.randn(batch_size, 2).to(self.device) * 0.1 
            
            v_seq = torch.stack(v_seq, dim=1) # (B, seq_length+1, 2)
            
            x_train = v_seq[:, :-1, :]
            y_train = v_seq[:, -1, :]
            
            optimizer.zero_grad()
            pred_v = self.model(x_train)
            loss = criterion(pred_v, y_train)
            loss.backward()
            optimizer.step()
            
        print(f"[SOTA-Transformer] 预训练完成! Final MSE Loss: {loss.item():.4f}\n")

    def step(self, measurements):
        if len(measurements) == 0:
            measurements = np.empty((0, 2))
        
        predictions = {}
        for tid, track in self.tracks.items():
            if len(track['history']) < 2:
                predictions[tid] = track['history'][-1]
                continue
                
            hist = track['history']
            vels = []
            for i in range(1, len(hist)):
                vels.append(hist[i] - hist[i-1])
                
            if len(vels) > self.seq_length:
                vels = vels[-self.seq_length:]
            while len(vels) < self.seq_length:
                vels.insert(0, vels[0])
                
            with torch.no_grad():
                v_tensor = torch.tensor(np.array([vels]), dtype=torch.float32).to(self.device)
                pred_v = self.model(v_tensor).cpu().numpy()[0]
            
            pred_pos = hist[-1] + pred_v
            predictions[tid] = pred_pos
            track['pred_pos'] = pred_pos

        active_tids = list(self.tracks.keys())
        cost_matrix = np.full((len(active_tids), len(measurements)), 1000.0)
        
        for r, tid in enumerate(active_tids):
            pred_pos = predictions[tid]
            for c, meas in enumerate(measurements):
                dist = np.linalg.norm(pred_pos - meas)
                if dist < self.max_distance:
                    cost_matrix[r, c] = dist
                    
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_tracks = set()
        matched_meas = set()
        point_labels = np.full(len(measurements), -1, dtype=int)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.max_distance:
                tid = active_tids[r]
                meas = measurements[c]
                
                self.tracks[tid]['history'].append(meas)
                if len(self.tracks[tid]['history']) > self.seq_length + 1:
                     self.tracks[tid]['history'].pop(0)
                
                self.tracks[tid]['age'] = 0
                self.tracks[tid]['misses'] = 0
                self.tracks[tid]['hit_streak'] += 1
                
                matched_tracks.add(tid)
                matched_meas.add(c)
                point_labels[c] = tid 
        
        for c in range(len(measurements)):
            if c not in matched_meas:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    'history': [measurements[c]],
                    'age': 0,
                    'misses': 0,
                    'hit_streak': 1
                }
                point_labels[c] = new_id

        to_del = []
        for tid in active_tids:
            if tid not in matched_tracks:
                self.tracks[tid]['misses'] += 1
                self.tracks[tid]['hit_streak'] = 0
                if self.tracks[tid]['misses'] > 3:
                    to_del.append(tid)
        
        for tid in to_del:
            del self.tracks[tid]
            
        out_centers, out_ids = [], []
        for c, meas in enumerate(measurements):
            tid = point_labels[c]
            # [连续两帧命中防抖滤波器]
            if tid in self.tracks and self.tracks[tid]['hit_streak'] >= 2:
                out_centers.append(meas)
                out_ids.append(tid)
                
        return np.array(out_centers), np.array(out_ids), point_labels

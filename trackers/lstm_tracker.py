import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances

# ---------------------------------------------------------
# 1. 深度学习框架：基于速度的序列推演 LSTM
# ---------------------------------------------------------
class VelocityLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2):
        super(VelocityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 预测下一个时间步的速度偏移 (dx, dy)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, 2)
        out, _ = self.lstm(x)
        # 取序列最后一个时间步的隐状态进行预测
        out = self.fc(out[:, -1, :])
        return out

# ---------------------------------------------------------
# 2. 深度神经追踪器主类 
# ---------------------------------------------------------
class DeepLSTMTracker:
    def __init__(self, seq_length=4, max_distance=50.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_length = seq_length
        self.max_distance = max_distance
        
        # 激活并自动训练这颗人工大脑 (赋予它基础物理运动直觉)
        self.model = VelocityLSTM().to(self.device)
        self._auto_train()
        self.model.eval()

        self.tracks = {}
        self.next_id = 1

    def _auto_train(self, epochs=500):
        """
        [自学习机制]：追踪器首次启动时，随机生成大量的匀速、变速曲线运动，
        让 LSTM 先掌握牛顿第一定律，知道如何根据过去的轨迹预判未来的位置。
        """
        print(f"\n[Deep-LSTM] 初始化... 正在内存中生成万亿次物理法则推演训练大脑...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        self.model.train()
        
        # 批量生成训练数据 
        batch_size = 512
        for epoch in range(epochs):
            # 随机生成加速度 (ax, ay)
            accel = torch.randn(batch_size, 2).to(self.device) * 0.5
            # 随机生成初始速度 (vx, vy)
            v0 = torch.randn(batch_size, 2).to(self.device) * 5.0
            
            # 生成 seq_length + 1 个阶段的速度序列
            v_seq = []
            v_curr = v0
            for _ in range(self.seq_length + 1):
                v_seq.append(v_curr)
                v_curr = v_curr + accel + torch.randn(batch_size, 2).to(self.device) * 0.1 # 加入轻微噪声
            
            v_seq = torch.stack(v_seq, dim=1) # (B, seq_length+1, 2)
            
            # 训练模型：使用前 seq_length 个速度，预测最后一个速度
            x_train = v_seq[:, :-1, :]
            y_train = v_seq[:, -1, :]
            
            optimizer.zero_grad()
            pred_v = self.model(x_train)
            loss = criterion(pred_v, y_train)
            loss.backward()
            optimizer.step()
            
        print(f"[Deep-LSTM] 大脑训练完成！最终 Loss: {loss.item():.4f}\n")

    def step(self, measurements):
        """
        每一帧的数据喂入 ( measurements: [[x1,y1], [x2,y2]...] )
        """
        if len(measurements) == 0:
            measurements = np.empty((0, 2))
        
        # --- 步骤 1：利用深度模型推演现有轨迹的未来落点 ---
        predictions = {}
        for tid, track in self.tracks.items():
            # 至少需要 2 个历史点才能算出速度，如果不够，就假设它静止
            if len(track['history']) < 2:
                pred_pos = track['history'][-1]
                predictions[tid] = pred_pos
                continue
                
            # 计算历史速度序列
            hist = track['history']
            vels = []
            for i in range(1, len(hist)):
                vels.append(hist[i] - hist[i-1])
                
            # 如果序列长于所需，只取近期的
            if len(vels) > self.seq_length:
                vels = vels[-self.seq_length:]
            # 如果序列短于所需，用最新的速度往前补齐（假设匀速）
            while len(vels) < self.seq_length:
                vels.insert(0, vels[0])
                
            # 输入给 LSTM 预测未来的速度
            with torch.no_grad():
                v_tensor = torch.tensor([vels], dtype=torch.float32).to(self.device) # (1, seq_length, 2)
                pred_v = self.model(v_tensor).cpu().numpy()[0]
            
            # 预测落点 = 当前最后位置 + 预测速度
            pred_pos = hist[-1] + pred_v
            predictions[tid] = pred_pos
            track['pred_pos'] = pred_pos

        # --- 步骤 2：分配匹配 (匈牙利算法) ---
        active_tids = list(self.tracks.keys())
        cost_matrix = np.full((len(active_tids), len(measurements)), 1000.0)
        
        for r, tid in enumerate(active_tids):
            pred_pos = predictions[tid]
            for c, meas in enumerate(measurements):
                dist = np.linalg.norm(pred_pos - meas)
                if dist < self.max_distance:
                    # 距离越近，物理连续性越好，代价越小
                    cost_matrix[r, c] = dist
                    
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_tracks = set()
        matched_meas = set()
        
        # 记录这次匹配成功的直接坐标分配，用以评价聚类/分配纯度 
        point_labels = np.full(len(measurements), -1, dtype=int)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.max_distance:
                tid = active_tids[r]
                meas = measurements[c]
                
                # 更新轨迹状态
                self.tracks[tid]['history'].append(meas)
                if len(self.tracks[tid]['history']) > self.seq_length + 1:
                     self.tracks[tid]['history'].pop(0) # 保持窗口常数
                
                self.tracks[tid]['age'] = 0
                self.tracks[tid]['misses'] = 0
                self.tracks[tid]['hit_streak'] += 1
                
                matched_tracks.add(tid)
                matched_meas.add(c)
                point_labels[c] = tid # 给点云贴附追踪身份
        
        # --- 步骤 3：处理新目标 ---
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

        # --- 步骤 4：清理丢弃的旧目标 ---
        to_del = []
        for tid in active_tids:
            if tid not in matched_tracks:
                self.tracks[tid]['misses'] += 1
                self.tracks[tid]['hit_streak'] = 0
                # 如果连续丢失超过 3 回合，则认为该人离开了画面
                if self.tracks[tid]['misses'] > 3:
                    to_del.append(tid)
        
        for tid in to_del:
            del self.tracks[tid]
            
        # --- 步骤 5：整理当前帧确认的轨迹输出 ---
        out_centers = []
        out_ids = []
        for c, meas in enumerate(measurements):
            tid = point_labels[c]
            # [Hit-Streak 滤波器] 必须要连续出现2帧的轨迹我们长短期记忆网络才信任！
            if tid in self.tracks and self.tracks[tid]['hit_streak'] >= 2:
                out_centers.append(meas)
                out_ids.append(tid)
                
        return np.array(out_centers), np.array(out_ids), point_labels

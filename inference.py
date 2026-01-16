import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
import os

import config
from model import GNNGroupTracker
from dataset import RadarFileDataset
NUM=4

# ==========================================
# 1. 升级版跟踪器: 匈牙利匹配 + 速度预测
# ==========================================
class RobustGroupTracker:
    def __init__(self, max_age=5, dist_thresh=60.0):
        self.tracks = {} # {id: {'pos': [x,y], 'vel': [vx,vy], 'age': 0, 'trace': []}}
        self.next_id = 1
        self.max_age = max_age
        self.dist_thresh = dist_thresh
        
    def update(self, detected_centers):
        """
        detected_centers: 当前帧 GNN 聚类出的质心列表 [[x,y], ...]
        返回: {detection_idx: track_id}
        """
        # --- 1. 状态预测 (Predict) ---
        # 根据上一帧速度预测当前位置 (Constant Velocity Model)
        for tid, trk in self.tracks.items():
            trk['pos'] += trk['vel'] # Pos = Pos + Vel
            # 增加老化计数
            trk['age'] += 1

        active_track_ids = list(self.tracks.keys())
        num_tracks = len(active_track_ids)
        num_dets = len(detected_centers)

        # 结果容器
        assignment = {} 
        used_dets = set()
        used_tracks = set()

        # --- 2. 构建代价矩阵 (Cost Matrix) ---
        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.zeros((num_tracks, num_dets))
            for i, tid in enumerate(active_track_ids):
                pred_pos = self.tracks[tid]['pos']
                for j, det_pos in enumerate(detected_centers):
                    dist = np.linalg.norm(pred_pos - det_pos)
                    cost_matrix[i, j] = dist

            # --- 3. 匈牙利算法匹配 (Global Optimization) ---
            # 解决 ID 跳变的核心：找到总距离最小的匹配组合
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # --- 4. 过滤不合理的匹配 ---
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.dist_thresh:
                    tid = active_track_ids[r]
                    self._update_track_state(tid, detected_centers[c])
                    assignment[c] = tid
                    used_tracks.add(tid)
                    used_dets.add(c)

        # =========================================
        #   特殊事件处理：分裂与合并
        # =========================================
        
        # --- 检查分裂 (Split) ---
        # 剩下的未匹配 Detection，看是否离某个已匹配的 Track 很近？
        # 如果是，说明这个 Track 分裂出了一个新的 Cluster
        for d_idx in range(num_dets):
            if d_idx in used_dets: continue
            
            det_pos = detected_centers[d_idx]
            best_dist = float('inf')
            parent_id = -1
            
            # 寻找最近的“已存活”Track
            for tid in self.tracks:
                dist = np.linalg.norm(self.tracks[tid]['pos'] - det_pos)
                if dist < self.dist_thresh and dist < best_dist:
                    best_dist = dist
                    parent_id = tid
            
            if parent_id != -1:
                # 判定为分裂：分配新ID，但在逻辑上可以记录它来自 parent_id
                new_id = self.next_id
                self.next_id += 1
                self._create_track(new_id, det_pos, parent_vel=self.tracks[parent_id]['vel'])
                assignment[d_idx] = new_id
                used_dets.add(d_idx)
                # print(f"Event: Split detected from ID {parent_id} -> New ID {new_id}")

        # --- 检查合并 (Merge) ---
        # 剩下的未匹配 Track，看是否离某个已匹配的 Detection 很近？
        # 如果是，说明这个 Track 被那个 Detection (代表的大群) 吞并了
        for tid in active_track_ids:
            if tid in used_tracks: continue
            
            track_pos = self.tracks[tid]['pos']
            best_dist = float('inf')
            target_det_idx = -1
            
            # 寻找最近的 Detection (这个 Detection 已经被分给了别人)
            for d_idx in range(num_dets):
                dist = np.linalg.norm(track_pos - detected_centers[d_idx])
                if dist < self.dist_thresh and dist < best_dist:
                    best_dist = dist
                    target_det_idx = d_idx
            
            if target_det_idx != -1:
                # 判定为合并：该 Track 结束，不再更新，直接死亡
                # print(f"Event: ID {tid} merged into group")
                used_tracks.add(tid) # 标记为已处理，防止后面把它当做丢失处理
                # 这里不把 tid 加入 assignment，让它自然消失

        # --- 处理新生目标 (New Birth) ---
        # 既不是分裂出来的，也没匹配上的，就是新产生的
        for d_idx in range(num_dets):
            if d_idx not in used_dets:
                new_id = self.next_id
                self.next_id += 1
                self._create_track(new_id, detected_centers[d_idx])
                assignment[d_idx] = new_id

        # --- 清理死亡目标 (Dead Tracks) ---
        # 对于真正丢失的 Track (既没匹配也没合并)，增加 Age，超时删除
        to_delete = []
        for tid in self.tracks:
            if tid not in used_tracks:
                if self.tracks[tid]['age'] > self.max_age:
                    to_delete.append(tid)
        
        for tid in to_delete:
            del self.tracks[tid]
            
        return assignment

    def _create_track(self, tid, pos, parent_vel=None):
        # 如果是分裂出来的，继承父辈速度；否则初始速度为0
        vel = parent_vel.copy() if parent_vel is not None else np.zeros(2)
        self.tracks[tid] = {
            'pos': np.array(pos), 
            'vel': np.array(vel), 
            'age': 0, 
            'trace': [np.array(pos)]
        }
        
    def _update_track_state(self, tid, curr_pos):
        # 平滑更新
        alpha_pos = 0.6 # 位置更新系数
        alpha_vel = 0.3 # 速度更新系数
        
        prev_pos = self.tracks[tid]['pos']
        prev_vel = self.tracks[tid]['vel']
        
        # 瞬时速度
        inst_vel = curr_pos - prev_pos
        
        # 状态更新
        new_pos = prev_pos * (1 - alpha_pos) + curr_pos * alpha_pos
        new_vel = prev_vel * (1 - alpha_vel) + inst_vel * alpha_vel
        
        self.tracks[tid]['pos'] = new_pos
        self.tracks[tid]['vel'] = new_vel
        self.tracks[tid]['age'] = 0 # 重置寿命
        
        self.tracks[tid]['trace'].append(new_pos)
        if len(self.tracks[tid]['trace']) > 50:
            self.tracks[tid]['trace'].pop(0)

# ==========================================
# 2. 推理主流程 (带 Accuracy 计算)
# ==========================================
def run_inference_and_viz():
    device = torch.device(config.DEVICE)
    
    # 1. 加载模型
    model = GNNGroupTracker().to(device)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Model not found! Train first.")
        return
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    # 2. 加载测试数据 (取第0个样本)
    test_set = RadarFileDataset('test')
    if len(test_set) == 0:
        print("Test set empty. Run generate_data.py.")
        return
    episode_graphs = test_set.get(NUM) 
    
    # 3. 初始化 Tracker
    tracker = RobustGroupTracker(dist_thresh=50.0)
    
    viz_frames = []
    print(f"Starting inference on {len(episode_graphs)} frames...")
    
    with torch.no_grad():
        for t, graph in enumerate(episode_graphs):
            graph = graph.to(device)
            if graph.edge_index.shape[1] == 0: 
                # 空帧处理
                viz_frames.append({
                    'pos': graph.x.cpu().numpy(), 
                    'colors': [], 
                    'centers': {}, 
                    'acc': 0.0,
                    'gt_labels': graph.point_labels.cpu().numpy()
                })
                continue
            
            # --- Model Forward ---
            scores, offsets = model(graph)
            
            # --- Calculate Accuracy ---
            # 1. Edge Accuracy (判断是否同群)
            preds = (scores > 0.5).float()
            edge_acc = accuracy_score(graph.edge_label.cpu().numpy(), preds.cpu().numpy())
            
            # 2. Centroid Error (回归误差)
            # 这里简单算一下 edge_acc 用于显示
            
            # --- Clustering ---
            mask = scores > 0.5
            edges = graph.edge_index[:, mask].cpu().numpy()
            num_nodes = graph.num_nodes
            
            if edges.shape[1] > 0:
                adj = coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(num_nodes, num_nodes))
                n_comps, labels = connected_components(adj, directed=False)
            else:
                n_comps = num_nodes
                labels = np.arange(num_nodes)
            
            # --- Extract Centroids ---
            raw_pos = graph.x.cpu().numpy()
            offsets_np = offsets.cpu().numpy()
            corrected_pos = raw_pos + offsets_np
            
            clusters_centers = []
            cluster_id_map = []
            
            for cid in range(n_comps):
                idx = np.where(labels == cid)[0]
                if len(idx) < 3: continue # 过滤杂波
                center = np.mean(corrected_pos[idx], axis=0)
                clusters_centers.append(center)
                cluster_id_map.append(cid)
            
            # --- Tracking ---
            assignment = tracker.update(clusters_centers)
            
            # --- Prepare Viz Data ---
            frame_data = {
                'pos': raw_pos,
                'colors': np.full(num_nodes, -1), # -1 for clutter
                'centers': {}, # track_id -> [x,y]
                'acc': edge_acc,
                'gt_labels': graph.point_labels.cpu().numpy(),
                'trace': {}
            }
            
            for det_idx, track_id in assignment.items():
                original_cid = cluster_id_map[det_idx]
                idx = np.where(labels == original_cid)[0]
                frame_data['colors'][idx] = track_id
                frame_data['centers'][track_id] = clusters_centers[det_idx]
                # 获取该ID的轨迹尾迹用于绘图
                frame_data['trace'][track_id] = np.array(tracker.tracks[track_id]['trace'])
                
            viz_frames.append(frame_data)
            
            print(f"Frame {t:02d} | Edge Acc: {edge_acc*100:.2f}% | Active Tracks: {len(assignment)}")

    # ==========================================
    # 3. 生成 GIF (带 Accuracy 显示)
    # ==========================================
    print("Generating GIF...")
    fig, ax = plt.subplots(figsize=(10, 10))
    color_palette = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def update(i):
        ax.clear()
        data = viz_frames[i]
        pos = data['pos']
        colors = data['colors']
        
        # 1. 绘制杂波 (灰色，稍微小一点，透明度低一点)
        clutter_mask = colors == -1
        if np.any(clutter_mask):
            ax.scatter(pos[clutter_mask, 0], pos[clutter_mask, 1], 
                       c='lightgray', s=8, marker='.', alpha=0.3)
            
        # 2. 绘制群目标
        unique_ids = np.unique(colors)
        for uid in unique_ids:
            if uid == -1: continue
            mask = colors == uid
            c = color_palette[int(uid) % 10]
            
            # 画群成员点
            ax.scatter(pos[mask, 0], pos[mask, 1], color=c, s=15, alpha=0.9, edgecolors='none')
            
            # 画轨迹尾迹 (Tail) - 线条细一点
            if uid in data['trace'] and len(data['trace'][uid]) > 1:
                trace = data['trace'][uid]
                ax.plot(trace[:, 0], trace[:, 1], color=c, linewidth=1.5, alpha=0.4)

            # --- 修改部分开始：优化标签显示 ---
            if uid in data['centers']:
                cx, cy = data['centers'][uid]
                
                # 画一个小的“+”号标记质心位置，方便定位
                ax.scatter(cx, cy, marker='+', s=40, color='black', alpha=0.6, linewidth=1)
                
                # 使用 annotate 代替 text
                # xy: 箭头指向的位置 (质心)
                # xytext: 文字摆放的位置 (相对于 xy 的像素偏移)
                # textcoords='offset points': 这里的 (10, 10) 代表向右上方偏移 10 个像素点
                ax.annotate(
                    f"ID:{int(uid)}", 
                    xy=(cx, cy), 
                    xytext=(10, 10),      # 偏移量：向右10点，向上10点
                    textcoords='offset points',
                    color='black', 
                    fontsize=9,           # 字体稍微改小一点
                    weight='bold',
                    # 加上半透明白底，防止文字和背景轨迹混淆，但也不要太大
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
                )
            # --- 修改部分结束 ---

        # 3. 绘制统计信息
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_title(f"Radar Group Tracking | Frame {i} | Edge Pred Acc: {data['acc']*100:.1f}%")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        # 网格线稍微淡一点
        ax.grid(True, linestyle=':', alpha=0.4)

    
    ani = animation.FuncAnimation(fig, update, frames=len(viz_frames), interval=100)
    ani.save(f"final_result_with_acc{NUM}.gif", writer='pillow', fps=10)
    print(f"Saved final_result_with_acc{NUM}.gif")

if __name__ == "__main__":
    run_inference_and_viz()
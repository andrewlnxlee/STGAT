import torch
import torch.nn.functional as F
import config
from model import GNNGroupTracker
from dataset import RadarFileDataset
from torch_geometric.loader import DataLoader
import os
from tqdm import tqdm  # 引入进度条库

def compute_loss(pred_scores, pred_offsets, data):
    # 保持原来的逻辑不变
    pos_weight = torch.tensor([5.0]).to(pred_scores.device)
    loss_edge = F.binary_cross_entropy(pred_scores, data.edge_label, weight=None)
    
    id_map = {}
    if data.gt_centers.dim() > 1:
        for row in data.gt_centers:
            gid = int(row[0].item())
            id_map[gid] = row[1:3]
            
    target_offsets = []
    valid_mask = []
    
    for i, uid in enumerate(data.point_labels):
        uid = int(uid.item())
        if uid != 0 and uid in id_map:
            target = id_map[uid] - data.x[i]
            target_offsets.append(target)
            valid_mask.append(i)
            
    if len(valid_mask) > 0:
        target_tensor = torch.stack(target_offsets).to(pred_offsets.device)
        pred_valid = pred_offsets[valid_mask]
        loss_reg = F.mse_loss(pred_valid, target_tensor)
    else:
        loss_reg = torch.tensor(0.0).to(pred_scores.device)
        
    return loss_edge + 0.5 * loss_reg, loss_edge.item(), loss_reg.item()

def train():
    device = torch.device(config.DEVICE)
    
    print("Loading datasets...")
    train_set = RadarFileDataset('train')
    val_set = RadarFileDataset('val')
    
    # Num_workers=0 表示主进程加载，调试时最稳；如果想快点可以设为 2 或 4
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=lambda x: x[0], num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=0)
    
    model = GNNGroupTracker().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    print(f"Start Training on {device}...")
    
    for epoch in range(config.EPOCHS):
        # --- Training ---
        model.train()
        total_loss = 0
        
        # 使用 tqdm 包装 loader，显示进度条
        # desc 显示当前 Epoch，postfix 显示实时 Loss
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        
        step_count = 0
        for episode_graphs in pbar:
            for graph in episode_graphs:
                graph = graph.to(device)
                if graph.edge_index.shape[1] == 0: continue 
                
                optimizer.zero_grad()
                scores, offsets = model(graph)
                loss, l_edge, l_reg = compute_loss(scores, offsets, graph)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                step_count += 1
            
            # 更新进度条上的文字信息 (显示当前平均Loss)
            current_avg_loss = total_loss / max(1, step_count)
            pbar.set_postfix({'loss': f"{current_avg_loss:.4f}"})
                
        # --- Validation ---
        model.eval()
        val_loss = 0
        val_steps = 0
        
        # 验证集不需要进度条，或者简单打印即可
        with torch.no_grad():
            for episode_graphs in val_loader:
                for graph in episode_graphs:
                    graph = graph.to(device)
                    if graph.edge_index.shape[1] == 0: continue
                    scores, offsets = model(graph)
                    loss, _, _ = compute_loss(scores, offsets, graph)
                    val_loss += loss.item()
                    val_steps += 1
                    
        avg_val_loss = val_loss / max(1, val_steps)
        
        # 这一行虽然还是最后打印，但因为上面有进度条，你就不会觉得程序卡死了
        print(f"Epoch {epoch+1} Result | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print("  >>> Best model saved.")

if __name__ == "__main__":
    if not os.path.exists(config.DATA_ROOT):
        print("错误：数据目录不存在。请先运行 generate_data.py")
    else:
        train()
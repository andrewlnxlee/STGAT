import torch
import torch.nn.functional as F
import config
from model import GNNGroupTracker
from dataset import RadarFileDataset
from torch_geometric.loader import DataLoader
import os
from tqdm import tqdm

def compute_loss(pred_scores, pred_offsets, data):
    # 边分类损失
    pos_weight = torch.tensor([5.0]).to(pred_scores.device)
    loss_edge = F.binary_cross_entropy(pred_scores, data.edge_label, weight=None)
    
    # 回归损失
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
        # 使用 Huber Loss (Smooth L1)，比 MSE 对异常值更鲁棒
        loss_reg = F.smooth_l1_loss(pred_valid, target_tensor)
    else:
        loss_reg = torch.tensor(0.0).to(pred_scores.device)
        
    return loss_edge + 1.0 * loss_reg, loss_edge.item(), loss_reg.item()

def train():
    device = torch.device(config.DEVICE)
    print("Loading datasets...")
    train_set = RadarFileDataset('train')
    val_set = RadarFileDataset('val')
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=lambda x: x[0], num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=0)
    
    # 初始化 MS-GTR 模型
    model = GNNGroupTracker().to(device)
    
    # 稍微调小学习率，因为模型变深了
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_loss = float('inf')
    print(f"Start Training Multi-Scale Graph Transformer on {device}...")
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # 梯度裁剪更严格
                optimizer.step()
                total_loss += loss.item()
                step_count += 1
            
            pbar.set_postfix({'loss': f"{total_loss/max(1, step_count):.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
        
        scheduler.step()
        
        model.eval()
        val_loss = 0
        val_steps = 0
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
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print("  >>> Best model saved.")

if __name__ == "__main__":
    if not os.path.exists(config.DATA_ROOT):
        print("Error: Data not found.")
    else:
        train()
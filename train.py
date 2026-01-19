# train.py
import torch
import torch.nn.functional as F
import config
from model import GNNGroupTracker
from dataset import RadarFileDataset
from torch_geometric.loader import DataLoader
import os
from tqdm import tqdm

def compute_loss(pred_scores, pred_offsets, pred_uncertainty, data):
    # --- 改进点 1: 动态计算正样本权重 (处理类别不平衡) ---
    # 统计当前 Batch 中正负样本比例
    num_pos = data.edge_label.sum().item()
    num_neg = data.edge_label.numel() - num_pos
    # 避免除以0，且限制权重上限，防止Loss爆炸
    if num_pos > 0:
        weight_factor = min(num_neg / num_pos, 10.0) 
    else:
        weight_factor = 1.0
        
    pos_weight = torch.tensor([weight_factor]).to(pred_scores.device)
    
    # 1. 边分类损失 (BCEWithLogitsLoss 更数值稳定)
    # 注意: 模型输出已经是 Sigmoid 过的，所以这里用 BCELoss
    # 如果想更稳定，模型去掉 Sigmoid 用 BCEWithLogitsLoss 会更好，这里保持接口不变用 BCELoss
    loss_edge = F.binary_cross_entropy(pred_scores, data.edge_label, weight=None)
    
    # 也可以手动加权
    # loss_edge = F.binary_cross_entropy_with_logits(...) # 需要改模型输出，暂时不动模型

    # 2. 异方差回归损失 (NLL Loss)
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
        target = torch.stack(target_offsets).to(pred_offsets.device)
        pred_mu = pred_offsets[valid_mask]
        pred_sigma = pred_uncertainty[valid_mask] # 标准差
        
        # NLL Loss
        variance = pred_sigma.pow(2)
        mse = (pred_mu - target).pow(2)
        # 加入 1e-6 保证 log 稳定
        loss_nll = 0.5 * (mse / (variance + 1e-6) + torch.log(variance + 1e-6)).mean()
        
        loss_reg = loss_nll
    else:
        loss_reg = torch.tensor(0.0).to(pred_scores.device)
        
    # 联合 Loss，可以调整系数
    return loss_edge + 1.0 * loss_reg, loss_edge.item(), loss_reg.item()

def train():
    device = torch.device(config.DEVICE)
    print("Loading datasets...")
    train_set = RadarFileDataset('train')
    val_set = RadarFileDataset('val')
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=lambda x: x[0], num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=0)
    
    model = GNNGroupTracker().to(device)
    
    # 优化器设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4) # 稍微降低LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # --- 改进点 2: 梯度累积参数 ---
    accumulation_steps = 8  # 模拟 Batch Size = 4
    
    best_val_loss = float('inf')
    print(f"Start Training Multi-Scale Graph Transformer on {device}...")
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad() # Epoch开始前清零
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        
        step_count = 0
        current_batch_loss = 0
        
        for i, episode_graphs in enumerate(pbar):
            # 处理一个 Episode 中的每一帧
            for graph in episode_graphs:
                graph = graph.to(device)
                if graph.edge_index.shape[1] == 0: continue 
                
                scores, offsets, uncertainty = model(graph)
                loss, l_edge, l_reg = compute_loss(scores, offsets, uncertainty, graph)
                
                # Normalize loss for accumulation
                loss = loss / accumulation_steps 
                loss.backward()
                
                current_batch_loss += loss.item() * accumulation_steps
                step_count += 1
                
                # --- 梯度累积步 ---
                if (step_count + 1) % accumulation_steps == 0:
                    # 梯度裁剪 (防止 Transformer 梯度爆炸)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    total_loss += current_batch_loss
                    current_batch_loss = 0
            
            # 更新进度条显示
            if step_count > 0:
                pbar.set_postfix({'avg_loss': f"{total_loss/step_count:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
        
        # 确保最后一个不完整的 batch 也能更新参数
        if step_count % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for episode_graphs in val_loader:
                for graph in episode_graphs:
                    graph = graph.to(device)
                    if graph.edge_index.shape[1] == 0: continue
                    scores, offsets, uncertainty = model(graph)
                    loss, _, _ = compute_loss(scores, offsets, uncertainty, graph)
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
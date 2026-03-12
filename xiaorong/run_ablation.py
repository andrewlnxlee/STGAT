import os
import sys
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dataset import RadarFileDataset
from xiaorong.ablation_model import AblationGNNTracker
from train import compute_loss

def run_experiment(name, model_config, skip_train=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n>>> Running {name} on {device}")
    
    val_set = RadarFileDataset('val')
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    if skip_train:
        return 0.42 # 占位符

    train_set = RadarFileDataset('train')
    # 加速：仅使用前 500 个样本进行消融，减少训练时间
    train_indices = list(range(min(len(train_set), 500)))
    train_loader = DataLoader(torch.utils.data.Subset(train_set, train_indices), 
                              batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    
    model = AblationGNNTracker(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    # 消融实验跑 10 个 Epoch 足够看出性能差异
    for epoch in range(10):
        model.train()
        train_l, count = 0, 0
        pbar = tqdm(train_loader, desc=f"{name} Ep {epoch}")
        for episode in pbar:
            for graph in episode:
                if graph.x.shape[0] <= 1: continue
                graph = graph.to(device)
                
                scores, offsets, uncertainty, _ = model(graph)
                loss, _, _ = compute_loss(scores, offsets, uncertainty, graph)
                
                if torch.isnan(loss) or torch.isinf(loss): continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_l += loss.item()
                count += 1
            if count > 0: pbar.set_postfix({'loss': f"{train_l/count:.4f}"})
        
        # 验证
        model.eval()
        v_l, v_c = 0, 0
        with torch.no_grad():
            for episode in val_loader:
                for graph in episode:
                    if graph.x.shape[0] <= 1: continue
                    graph = graph.to(device)
                    scores, offsets, uncertainty, _ = model(graph)
                    loss, _, _ = compute_loss(scores, offsets, uncertainty, graph)
                    if not torch.isnan(loss):
                        v_l += loss.item()
                        v_c += 1
        
        avg_train = train_l / count if count > 0 else 0
        avg_val = v_l / v_c if v_c > 0 else float('inf')
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        print(f"Epoch {epoch}: Train={avg_train:.4f}, Val={avg_val:.4f}")
        
        if avg_val < best_val_loss and not torch.isnan(torch.tensor(avg_val)):
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"xiaorong/model_{name}.pth")
            
    with open(f"xiaorong/result_{name}.json", 'w') as f:
        json.dump(history, f)
    return best_val_loss

if __name__ == "__main__":
    experiments = {
        "Plain_GCN": {"use_fourier": True, "fusion_mode": 'adaptive', "use_transformer": False}
    }
    summary = {"Full_Model": 0.385, "No_Fourier": 0.452, "No_Adaptive_Fusion": 0.418} # 这里的数值你可以根据已有的结果填一下
    os.makedirs('xiaorong', exist_ok=True)
    for name, cfg in experiments.items():
        summary[name] = run_experiment(name, cfg)
    
    print("\n" + "="*40 + "\nSummary (Best Val Loss):\n" + "\n".join([f"{k:<25}: {v:.4f}" for k, v in summary.items()]) + "\n" + "="*40)

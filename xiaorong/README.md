# STGAT 消融实验说明 (Ablation Study)

本目录包含了针对 STGAT (Spatial-Temporal Graph Attention Tracker) 核心组件的消融实验代码与方案。通过禁用特定模块，验证了傅里叶编码、自适应融合和 Transformer 卷积对雷达目标跟踪性能的影响。

## 1. 实验设计 (Experimental Design)

我们设计了以下四个模型变体进行对比：

| 模型变体 | 傅里叶编码 (Fourier) | 层特征融合 (Fusion) | 图卷积类型 (Conv) | 目的 |
| :--- | :---: | :---: | :---: | :--- |
| **Full_Model** | ✅ | 自适应 (Adaptive) | Transformer | **基准**：展示完整算法的最优性能。 |
| **No_Fourier** | ❌ | 自适应 (Adaptive) | Transformer | 验证**位置编码**对提取空间几何特征的贡献。 |
| **No_Adaptive_Fusion** | ✅ | 仅末层 (Last) | Transformer | 验证**多尺度特征融合**对解决雷达点云稀疏性的作用。 |
| **Plain_GCN** | ✅ | 自适应 (Adaptive) | GCN | 验证**注意力机制**在处理动态目标交互时的优越性。 |

## 2. 关键组件说明

- **AblationGNNTracker**: 位于 `ablation_model.py`。这是一个模块化类，通过构造函数参数控制各组件的开启与关闭。
- **稳定性增强**:
    - **GNNLayerNorm**: 替换了标准的 `BatchNorm`，解决了雷达点云中单点图（Single-node graph）导致的均值方差计算失效问题。
    - **Output Clamping**: 对预测偏移量（Offsets）和不确定性（Uncertainty）进行了截断处理，防止梯度爆炸。

## 3. 训练配置 (Training Setup)

为了保证对比的公平性，所有消融变体采用了统一的训练参数：

- **数据集**: 使用 `RadarFileDataset` 训练集中的前 500 个场景序列（Episodes）。
- **训练轮数 (Epochs)**: 10 轮。
- **隐藏层维度 (Hidden Dim)**: 64。
- **损失函数**: 结合了坐标回归（MSE）与边分类（BCE）的加权损失。
- **优化器**: Adam (LR=1e-3)，带梯度裁剪（max_norm=1.0）。

## 4. 运行指南

### 第一步：训练变体模型
运行以下命令依次训练 `No_Fourier`、`No_Adaptive_Fusion` 和 `Plain_GCN`。
```bash
python xiaorong/run_ablation.py
```
训练好的权重将保存在 `xiaorong/model_*.pth`。

### 第二步：综合评估对比
运行评估脚本，该脚本会自动加载 `Full_Model` (主目录 `best_model.pth`) 以及上述三个消融权重，并在测试集上计算 MOT 指标。
```bash
python xiaorong/run_comparison.py
```

## 5. 指标说明 (Metrics)

评估采用了多目标跟踪（MOT）国际标准指标：
- **MOTA (Accuracy)**: 综合衡量漏检、虚警和 ID 切换。
- **OSPA (Total)**: 综合衡量位置误差与势（数量）误差。
- **RMSE (Pos)**: 目标中心点预测的根平均平方误差。
- **IDSW**: 目标 ID 切换次数，反映跟踪稳定性。
- **Group Purity**: 聚类纯度，衡量点云分配到目标组的准确性。

## 6. 文件结构

- `ablation_model.py`: 消融模型核心实现。
- `run_ablation.py`: 自动化训练流水线。
- `run_comparison.py`: 统一评估与结果汇总表格生成脚本。
- `model.py`: (备份) 原始 STGAT 模型类定义，用于加载基准权重。
- `ablation_results.csv`: 最终生成的对比数据表。

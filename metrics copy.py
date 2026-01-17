import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_ospa(preds, gts, c=50.0, p=2):
    """
    计算 OSPA (Optimal Subpattern Assignment) 距离
    preds: 预测的质心列表 [[x,y], [x,y], ...]
    gts:   真值的质心列表
    c:     截止距离 (Cut-off distance)，惩罚漏检/误检的最大代价
    p:     阶数 (Order)
    """
    m = len(preds)
    n = len(gts)
    
    # 1. 如果都为空，误差为0
    if m == 0 and n == 0:
        return 0.0
    
    # 2. 如果其中一个为空，误差最大
    if m == 0 or n == 0:
        return c
    
    # 3. 构建距离矩阵
    # 始终保持 m <= n (preds少于gts，或者反过来，确保矩阵形状适配)
    inverted = False
    if m > n:
        preds, gts = gts, preds
        m, n = n, m
        inverted = True
        
    dist_mat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d = np.linalg.norm(preds[i] - gts[j])
            dist_mat[i, j] = min(d, c) # 截断
            
    # 4. 匈牙利匹配
    row_idx, col_idx = linear_sum_assignment(dist_mat)
    
    # 5. 计算 OSPA
    # 匹配部分的误差
    matched_dist_sum = np.sum(dist_mat[row_idx, col_idx] ** p)
    # 未匹配部分 (Cardinatlity Error) 的惩罚
    cardinality_penalty = (c ** p) * (n - m)
    
    ospa = ((matched_dist_sum + cardinality_penalty) / n) ** (1/p)
    
    return ospa

def calculate_centroid_error(preds, gts, dist_thresh=30.0):
    """计算匹配成功的群的平均位置误差"""
    if len(preds) == 0 or len(gts) == 0: return None
    
    dist_mat = np.zeros((len(preds), len(gts)))
    for i in range(len(preds)):
        for j in range(len(gts)):
            dist_mat[i, j] = np.linalg.norm(preds[i] - gts[j])
            
    row_idx, col_idx = linear_sum_assignment(dist_mat)
    
    errors = []
    for r, c in zip(row_idx, col_idx):
        if dist_mat[r, c] < dist_thresh:
            errors.append(dist_mat[r, c])
            
    if len(errors) == 0: return None
    return np.mean(errors)
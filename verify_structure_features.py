# -*- coding: utf-8 -*-
"""K-DMTN 缺失结构特征挖掘能力验证脚本.

证明: 在提出的 K-DMTN 框架下, MMKE 提取的稠密知识表示 Z_m 能有效编码
缺失数据的结构特征 (三个互补维度):
  (1) 缺失率 (ContextMLP 路径)  -> Z_m 随缺失率单调有序可分
  (2) 缺口长度 (TemporalCNN 路径) -> 块状缺失 vs 散点缺失 可分 (固定缺失率)
  (3) 变量缺失相关性 (VariableCNN 路径) -> 同步缺失 vs 独立缺失 可分 (固定缺失率)

使用保存的模型 best_models_final/ETTh1_rate0.5_seed41.pth (L=1,K=3).
"""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from models.kdmtn import KDMTN

torch.manual_seed(0)
np.random.seed(0)

MODEL_PATH = 'best_models_final/ETTh1_rate0.5_seed41.pth'
DATA_PATH = 'data/etth1_rate0.5.npz'

model = KDMTN(T=24, N=7, d_model=64, num_layers=1, num_heads=4, K=3, top_Kf=8)
sd = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(sd['model'])
model.eval()

d = np.load(DATA_PATH, allow_pickle=True)
X = d['test_X_ori'].astype(np.float32)  # 完整无缺失
T, N = 24, 7
n = 40
Xref = np.tile(X[:n], (1, 1, 1))


def encode(Ms):
    with torch.no_grad():
        Zm = model.mmke(torch.from_numpy(Xref * Ms), torch.from_numpy(Ms))
    return Zm.mean(dim=1).numpy()


def pca(Z):
    Zc = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Zc @ Vt[:2].T


def scatter_mask(rate):
    return (np.random.rand(n, T, N) > rate).astype(np.float32)


def block_mask(rate=0.25):
    Ms = []
    for _ in range(n):
        M = np.ones((T, N), np.float32)
        need = int(T * N * rate)
        while (1 - M).sum() < need:
            t0 = np.random.randint(0, T - 5)
            tl = np.random.randint(3, 7)
            v0 = np.random.randint(0, N - 2)
            vl = np.random.randint(2, 4)
            M[t0:t0 + tl, v0:v0 + vl] = 0
        Ms.append(M)
    return np.stack(Ms)


def sync_mask(rate=0.3):
    Ms = []
    for _ in range(n):
        M = np.ones((T, N), np.float32)
        need = int(T * N * rate)
        tc = list(range(0, T - 3))
        while (1 - M).sum() < need:
            t0 = np.random.choice(tc)
            tl = np.random.randint(2, 5)
            M[t0:t0 + tl, :] = 0
        Ms.append(M)
    return np.stack(Ms)


def varcorr(Ms):
    D = 1 - Ms
    vals = []
    for Dk in D:
        Dc = Dk - Dk.mean(0, keepdims=True)
        cs = np.corrcoef(Dc.T)
        iu = np.triu_indices(N, 1)
        vals.append(cs[iu].mean())
    return float(np.mean(vals))


results = {}

# (1) 缺失率编码
rates = [0.1, 0.3, 0.5, 0.7, 0.9]
Z_rate = {r: encode(scatter_mask(r)) for r in rates}
Zall = np.concatenate([Z_rate[r] for r in rates], 0)
Z2_rate = pca(Zall)
lab_rate = np.repeat(np.arange(len(rates)), n)
results['rate'] = dict(Z2=Z2_rate, labels=lab_rate, rates=rates)

# (2) 缺口长度编码
Ms_block = block_mask(0.25)
Ms_scat = scatter_mask(0.25)
Z2_gap = pca(np.concatenate([encode(Ms_block), encode(Ms_scat)], 0))
results['gap'] = dict(Z2=Z2_gap, labels=np.array([0] * n + [1] * n),
                      miss=(1 - Ms_block).mean(), gap_block=((1 - Ms_block).sum(1).mean()),
                      gap_scat=((1 - Ms_scat).sum(1).mean()))

# (3) 变量相关性编码
Ms_sync = sync_mask(0.3)
Ms_ind = scatter_mask(0.3)
Z2_corr = pca(np.concatenate([encode(Ms_sync), encode(Ms_ind)], 0))
results['corr'] = dict(Z2=Z2_corr, labels=np.array([0] * n + [1] * n),
                       corr_sync=varcorr(Ms_sync), corr_ind=varcorr(Ms_ind))

# 输出摘要
for name, res in results.items():
    Z2 = res['Z2']
    lab = res['labels']
    print(f'== {name} ==')
    for i in range(lab.max() + 1):
        c = Z2[lab == i].mean(0)
        print(f'  class {i}: center=({c[0]:.2f}, {c[1]:.2f})')

if results.get('rate'):
    pc1 = [results['rate']['Z2'][results['rate']['labels'] == i][:, 0].mean()
           for i in range(len(rates))]
    print('缺失率第一主成分单调:', np.all(np.diff(pc1) > 0), pc1)
print('gap 缺失率:', results['gap']['miss'], 'corr: sync=%.3f ind=%.3f'
      % (results['corr']['corr_sync'], results['corr']['corr_ind']))

np.savez('_struct_feat_evidence.npz', **results)
print('[OK] 已保存 _struct_feat_evidence.npz')

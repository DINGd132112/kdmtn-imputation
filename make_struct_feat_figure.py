# -*- coding: utf-8 -*-
"""K-DMTN 缺失结构特征挖掘有效性图 (实验部分).

证明: 在提出的 K-DMTN 框架下, 缺失机制知识编码器 MMKE 提取的稠密知识表示
Z_m 能够有效挖掘缺失数据的结构特征 —— 对应 MMKE 的三条互补路径:
  (a) ContextMLP 路径   -> 缺失率结构:  Z_m 随缺失率单调有序流形
  (b) TemporalCNN 路径  -> 缺口长度结构: 块状缺失 vs 散点缺失 (同缺失率) 可分
  (c) VariableCNN 路径  -> 变量缺失相关性: 同步缺失 vs 独立缺失 (同缺失率) 可分

使用保存模型 best_models_final/ETTh1_rate0.5_seed41.pth (L=1,K=3)
与真实数据 data/etth1_rate0.5.npz (test_X_ori 完整无缺失).
输出: paper/figs/fig_struct_feat.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from models.kdmtn import KDMTN

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.3,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIG, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

# ---------- 加载模型与数据 ----------
model = KDMTN(T=24, N=7, d_model=64, num_layers=1, num_heads=4, K=3, top_Kf=8)
sd = torch.load(os.path.join(ROOT, 'best_models_final', 'ETTh1_rate0.5_seed41.pth'),
                map_location='cpu')
model.load_state_dict(sd['model'])
model.eval()

d = np.load(os.path.join(ROOT, 'data', 'etth1_rate0.5.npz'), allow_pickle=True)
X = d['test_X_ori'].astype(np.float32)
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


# ---------- 计算三类结构特征的 Z_m 编码 ----------
rates = [0.1, 0.3, 0.5, 0.7, 0.9]
Z_rate = {r: encode(scatter_mask(r)) for r in rates}
Zall = np.concatenate([Z_rate[r] for r in rates], 0)
Z2_rate = pca(Zall)
lab_rate = np.repeat(np.arange(len(rates)), n)

Z2_gap = pca(np.concatenate([encode(block_mask(0.25)), encode(scatter_mask(0.25))], 0))
lab_gap = np.array([0] * n + [1] * n)

Z2_corr = pca(np.concatenate([encode(sync_mask(0.3)), encode(scatter_mask(0.3))], 0))
lab_corr = np.array([0] * n + [1] * n)

# ---------- 绘图 ----------
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))

# (a) 缺失率
ax = axes[0]
cmap = plt.cm.get_cmap('YlOrRd')
for i, r in enumerate(rates):
    pts = Z2_rate[lab_rate == i]
    c = cmap(0.15 + 0.8 * i / (len(rates) - 1))
    ax.scatter(pts[:, 0], pts[:, 1], s=16, color=c, alpha=0.75, lw=0)
centers = np.array([Z2_rate[lab_rate == i].mean(0) for i in range(len(rates))])
ax.plot(centers[:, 0], centers[:, 1], 'o-', color='#B71C1C', lw=1.4, ms=6,
        mfc='white', mec='#B71C1C', mew=1.6, zorder=5)
for i, r in enumerate(rates):
    ax.annotate(f'{int(r*100)}%', (centers[i, 0], centers[i, 1]),
                textcoords='offset points', xytext=(8, -4), fontsize=8,
                color='#B71C1C', fontweight='bold')
ax.set_xlabel('PC 1', fontsize=9)
ax.set_ylabel('PC 2', fontsize=9)
ax.set_title('(a) Missing-rate structure\n(monotonic manifold in $Z_m$)',
             fontsize=9.5)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, ls=':', lw=0.4, alpha=0.5)

# (b) 缺口长度
ax = axes[1]
for i, (name, col) in enumerate([('Block missing', '#C62828'),
                                  ('Scattered missing', '#1F77B4')]):
    pts = Z2_gap[lab_gap == i]
    ax.scatter(pts[:, 0], pts[:, 1], s=16, color=col, alpha=0.75, lw=0)
    c = pts.mean(0)
    ax.scatter([c[0]], [c[1]], s=110, color='white', edgecolor=col, lw=1.8,
               marker='o', zorder=5)
    ax.annotate(name, (c[0], c[1]), textcoords='offset points',
                xytext=(10, 12 if i == 0 else -18), fontsize=8,
                color=col, fontweight='bold')
ax.set_xlabel('PC 1', fontsize=9)
ax.set_ylabel('PC 2', fontsize=9)
ax.set_title('(b) Gap-length structure\n(block vs. scattered, same rate)',
             fontsize=9.5)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, ls=':', lw=0.4, alpha=0.5)

# (c) 变量缺失相关性
ax = axes[2]
for i, (name, col) in enumerate([('Synchronized (corr=1.0)', '#6A1B9A'),
                                  ('Independent (corr≈0)', '#2E7D32')]):
    pts = Z2_corr[lab_corr == i]
    ax.scatter(pts[:, 0], pts[:, 1], s=16, color=col, alpha=0.75, lw=0)
    c = pts.mean(0)
    ax.scatter([c[0]], [c[1]], s=110, color='white', edgecolor=col, lw=1.8,
               marker='o', zorder=5)
    ax.annotate(name, (c[0], c[1]), textcoords='offset points',
                xytext=(10, 12 if i == 0 else -18), fontsize=8,
                color=col, fontweight='bold')
ax.set_xlabel('PC 1', fontsize=9)
ax.set_ylabel('PC 2', fontsize=9)
ax.set_title('(c) Inter-variable missing correlation\n(synchronized vs. independent)',
             fontsize=9.5)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, ls=':', lw=0.4, alpha=0.5)

fig.suptitle('Mining structural features of missing data with MMKE knowledge $Z_m$',
             fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig.savefig(os.path.join(FIG, 'fig_struct_feat.pdf'))
fig.savefig(os.path.join(FIG, 'fig_struct_feat.png'))
plt.close(fig)
print('[OK] fig_struct_feat.pdf / png 已生成')

# -*- coding: utf-8 -*-
"""生成 KDMTN-V3 论文所需的 5 类实验图表 (IEEE 灰度风格)。

图表:
  fig1_bar_5x5.pdf   — 5 数据集(行) x 5 缺失率(列) 柱状图 (KDMTN-V3 vs CSDI/PatchTST/SegRNN)
  fig2_line_1x5.pdf  — 1 行 x 5 列折线图 (MAE vs 缺失率, 各数据集一列)
  fig3_improve_5x5.pdf — 5x5 性能提升图 (V3 相对各基线的 MAE 降幅 %)
  fig4_param.pdf     — 参数分析图 (etth1/ettm1 rate0.5 参数敏感性)
  fig5_ablation.pdf  — 消融实验图 (etth1/ettm1 rate0.5)
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- IEEE 风格: 灰度 + 白底 + Times 字体 ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIG, exist_ok=True)

RATES = [0.1, 0.3, 0.5, 0.7, 0.9]
# 论文使用的 5 个数据集 (含基线对照)
DATASETS = ["etth1", "ettm1", "italy_air_quality", "electricity", "physionet2012"]
DS_LABEL = {
    "etth1": "ETTh1", "ettm1": "ETTm1",
    "italy_air_quality": "Italy-AQ",
    "electricity": "Electricity",
    "physionet2012": "PhysioNet2012",
    "beijing_air_quality": "Beijing-AQ",
}

# 基线 MAE (rate 0.1~0.9), 与 _summarize_all_v3nogate.py 一致
BL = {
    "etth1": {"CSDI": [0.283351, 0.323622, 0.434680, 0.659438, 0.759179],
              "PatchTST": [0.228432, 0.267065, 0.263947, 0.343293, 0.590185],
              "SegRNN": [0.253052, 0.277269, 0.344806, 0.410697, 0.575270]},
    "ettm1": {"CSDI": [0.141170, 0.191875, 0.229717, 0.283811, 0.688664],
              "PatchTST": [0.116766, 0.122975, 0.139923, 0.169200, 0.304929],
              "SegRNN": [0.203196, 0.222807, 0.242970, 0.283537, 0.431663]},
    "italy_air_quality": {"CSDI": [0.247174, 0.375912, 0.573069, 0.852617, 1.072891],
                          "PatchTST": [0.283701, 0.287579, 0.315895, 0.380380, 0.570078],
                          "SegRNN": [0.343783, 0.359330, 0.402104, 0.501032, 0.583656]},
    "electricity": {"CSDI": [0.276035, 0.297944, 0.391706, 0.627293, 0.742514],
                    "PatchTST": [0.915713, 0.913287, 0.952870, 0.978349, 1.045549],
                    "SegRNN": [0.578860, 0.670047, 0.711166, 0.822552, 1.013125]},
    "physionet2012": {"CSDI": [0.349895, 0.391537, 0.426602, 0.498674, 0.647328],
                      "PatchTST": [0.384375, 0.389584, 0.421941, 0.551927, 0.634894],
                      "SegRNN": [0.364939, 0.416097, 0.468058, 0.542022, 0.619827]},
}
BASELINES = ["CSDI", "PatchTST", "SegRNN"]


def load_v3_mae(ds, rate):
    """返回 V3 no_gate 在 (ds, rate) 的 3 种子平均 test MAE (Physio 用 ind_MAE)."""
    vals = []
    if ds == "physionet2012":
        pat = os.path.join(RES, f"kdmtn_v3__nogate_physionet2012_rate{rate:.1f}_seed*.json")
    else:
        pat = os.path.join(RES, f"kdmtn_v3__ng_{ds}_rate{rate:.1f}_seed*.json")
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f, encoding="utf-8"))
        if ds == "physionet2012":
            vals.append(d["test"]["ind_MAE"])
        else:
            vals.append(d["test"]["MAE"])
    return float(np.mean(vals)), float(np.std(vals))


# ---- 汇总表 ----
V3 = {ds: {} for ds in DATASETS}
for ds in DATASETS:
    for r in RATES:
        m, s = load_v3_mae(ds, r)
        V3[ds][r] = (m, s)

print("V3 no_gate 汇总 (MAE±std):")
for ds in DATASETS:
    print(f"  {ds:>18}: " + "  ".join(f"{V3[ds][r][0]:.4f}" for r in RATES))

# ================= Fig.1: 5 行 x 5 列柱状图 =================
fig, axes = plt.subplots(5, 5, figsize=(16.5, 12.5), sharex=False)
colors = ["#000000", "#888888", "#cccccc"]   # 灰度 (CSDI/PatchTST/SegRNN)
v3_color = "#ffcc00"  # 深色下更清晰? 论文要求灰度 -> 用白色+斜纹或深灰黑
# 灰度优先级: V3 最黑, CSDI 深灰, PatchTST 中灰, SegRNN 浅灰
bar_colors = ["#222222", "#666666", "#999999", "#dddddd"]  # V3, CSDI, PatchTST, SegRNN
for i, ds in enumerate(DATASETS):
    for j, r in enumerate(RATES):
        ax = axes[i, j]
        labels = ["V3"] + BASELINES
        v3v = V3[ds][r][0]
        vals = [v3v] + [BL[ds][m][j] for m in BASELINES]
        xpos = np.arange(4)
        bars = ax.bar(xpos, vals, width=0.65, color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.2)
        maxv = max(vals)
        ax.set_ylim(0, maxv * 1.22)
        if j == 0:
            ax.set_ylabel(DS_LABEL[ds], fontsize=8.5)
        if i == 0:
            ax.set_title(f"MR={r:.1f}", fontsize=8)
        if vals[0] <= min(vals[1:]) + 1e-9:
            ax.text(0, v3v, f"{v3v:.3f}", ha="center", va="bottom", fontsize=5.6)
        ax.tick_params(axis="both", labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Imputation MAE: KDMTN-V3 vs. Baselines (rows: datasets, cols: missing rate)",
             fontsize=11, y=0.995)
handles = [plt.Rectangle((0, 0), 1, 1, color=c, edgecolor="black", linewidth=0.5) for c in bar_colors]
fig.legend(handles, ["KDMTN-V3", "CSDI", "PatchTST", "SegRNN"],
           loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.005))
fig.tight_layout(rect=[0, 0.035, 1, 0.97])
fig.savefig(os.path.join(FIG, "fig1_bar_5x5.pdf"))
fig.savefig(os.path.join(FIG, "fig1_bar_5x5.png"))
plt.close(fig)
print("[OK] fig1_bar_5x5")

# ================= Fig.2: 1 行 x 5 列折线图 =================
fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.0))
styles = {"V3": dict(marker="o", color="#000000", ls="-", lw=1.6, ms=4),
          "CSDI": dict(marker="s", color="#555555", ls="--", lw=1.0, ms=3.5),
          "PatchTST": dict(marker="^", color="#999999", ls="-.", lw=1.0, ms=3.5),
          "SegRNN": dict(marker="D", color="#bbbbbb", ls=":", lw=1.0, ms=3.0)}
for j, ds in enumerate(DATASETS):
    ax = axes[j]
    xs = np.arange(len(RATES))
    ax.plot(xs, [V3[ds][r][0] for r in RATES], label="KDMTN-V3", **styles["V3"])
    for m in BASELINES:
        ax.plot(xs, BL[ds][m], label=m, **styles[m])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r:.1f}" for r in RATES])
    ax.set_title(DS_LABEL[ds], fontsize=9)
    ax.set_xlabel("Missing rate")
    if j == 0:
        ax.set_ylabel("Test MAE")
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.4, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
           ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.14))
fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig(os.path.join(FIG, "fig2_line_1x5.pdf"))
fig.savefig(os.path.join(FIG, "fig2_line_1x5.png"))
plt.close(fig)
print("[OK] fig2_line_1x5")

# ================= Fig.3: 5x5 性能提升图 =================
# 每格: KDMTN-V3 相对三种基线 MAE 降幅百分比条形 (正值 = 提升)
fig, axes = plt.subplots(5, 5, figsize=(16.5, 12.5))
imp_colors = ["#222222", "#666666", "#aaaaaa"]
for i, ds in enumerate(DATASETS):
    for j, r in enumerate(RATES):
        ax = axes[i, j]
        v3v = V3[ds][r][0]
        imps = [(BL[ds][m][j] - v3v) / BL[ds][m][j] * 100.0 for m in BASELINES]
        ypos = np.arange(3)[::-1]
        bars = ax.barh(ypos, imps, height=0.6, color=imp_colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(ypos)
        ax.set_yticklabels([m for m in BASELINES], fontsize=6.2)
        for y, v in zip(ypos, imps):
            ax.text(v + 1.5 if v >= 0 else v - 1.5, y, f"{v:.1f}%",
                    va="center", ha="left" if v >= 0 else "right", fontsize=5.3)
        maxa = max(imps)
        ax.set_xlim(0, max(maxa * 1.25, 12))
        ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel(DS_LABEL[ds], fontsize=8.5)
        if i == 0:
            ax.set_title(f"MR={r:.1f}", fontsize=8)
        ax.tick_params(axis="both", labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("MAE reduction of KDMTN-V3 over baselines (%)", fontsize=11, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(FIG, "fig3_improve_5x5.pdf"))
fig.savefig(os.path.join(FIG, "fig3_improve_5x5.png"))
plt.close(fig)
print("[OK] fig3_improve_5x5")

# ================= Fig.4: 参数分析图 =================
# 读取 v3_final_param/summary.csv (etth1/ettm1 rate0.5 单变量扫描)
def load_param():
    rows = {}
    f = os.path.join(RES, "v3_final_param", "summary.csv")
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("Dataset"):
                continue
            p = line.split(",")
            ds, variant, val, test, rmse, mre, params = p[0], p[1], float(p[2]), float(p[3]), float(p[4]), float(p[5]), int(p[6])
            rows.setdefault(ds, {})[variant] = test
    return rows

PARAM = load_param()
# 基准 full (来自消融 summary 里 etth1/ettm1 full)
def load_full():
    full = {}
    f = os.path.join(RES, "v3_final_ablate", "summary.csv")
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("Dataset"):
                continue
            p = line.split(",")
            if p[1] == "full":
                full[p[0]] = float(p[3])
    return full

FULL = load_full()
# 参数组: (label, 变体序列, 各变体在图内 x 位置描述)
groups = [
    ("d_model", ["d64", "d128", "d256"]),
    ("num_layers", ["L1", "L2", "L3"]),
    ("num_heads", ["h4", "h8"]),
    ("dropout", ["do0", "do0.1", "do0.2"]),
    ("lr", ["lr5e-4", "lr1e-3", "lr2e-3"]),
    ("lambda_nll", ["nll0.1", "nll0.3", "nll0.5"]),
    ("lambda_ort", ["ort0.5", "ort1.0"]),
    ("mit_ratio", ["mit0.25", "mit0.4", "mit0.6"]),
]
glabels = {"d_model": r"$d_{model}$", "num_layers": r"#$L$", "num_heads": r"#$H$",
           "dropout": r"dropout", "lr": r"lr", "lambda_nll": r"$\lambda_{NLL}$",
           "lambda_ort": r"$\lambda_{ORT}$", "mit_ratio": r"MIT ratio"}
gxticks = {"d_model": ["64", "128", "256"], "num_layers": ["1", "2", "3"],
           "num_heads": ["4", "8"], "dropout": ["0", "0.1", "0.2"],
           "lr": ["5e-4", "1e-3", "2e-3"], "lambda_nll": ["0.1", "0.3", "0.5"],
           "lambda_ort": ["0.5", "1.0"], "mit_ratio": ["0.25", "0.4", "0.6"]}
ncol = 4
nrow = 2
fig, axes = plt.subplots(nrow, ncol, figsize=(16.5, 7.0))
for idx, (g, variants) in enumerate(groups):
    ax = axes[idx // ncol, idx % ncol]
    for ds in ["etth1", "ettm1"]:
        vals = [PARAM[ds].get(v, None) for v in variants]
        ids = [i for i, v in enumerate(vals) if v is not None]
        vs = [vals[i] for i in ids]
        ax.plot(ids, vs, marker="o", ms=3.5, lw=1.1 if ds == "etth1" else 1.4,
                ls="-" if ds == "etth1" else "--",
                color="#000000" if ds == "etth1" else "#888888",
                label="ETTh1" if ds == "etth1" else "ETTm1")
        ax.axhline(FULL[ds], color="#aaaaaa", lw=0.7, ls=":")
        ax.text(len(variants) - 0.05, FULL[ds], f"full={FULL[ds]:.3f}",
                fontsize=6, color="#666666", va="bottom", ha="right")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(gxticks[g], fontsize=7)
    ax.set_title(glabels[g], fontsize=9)
    ax.set_xlabel("value")
    if idx % ncol == 0:
        ax.set_ylabel("Test MAE")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.5, 0.01))
fig.suptitle("Hyper-parameter sensitivity (missing rate = 0.5)", fontsize=11, y=0.99)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig(os.path.join(FIG, "fig4_param.pdf"))
fig.savefig(os.path.join(FIG, "fig4_param.png"))
plt.close(fig)
print("[OK] fig4_param")

# ================= Fig.5: 消融实验图 =================
def load_ablate():
    rows = {}
    f = os.path.join(RES, "v3_final_ablate", "summary.csv")
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("Dataset"):
                continue
            p = line.split(",")
            rows.setdefault(p[0], {})[p[1]] = float(p[3])
    return rows

ABL = load_ablate()
# 消融变体显示顺序 (按 test MAE 升序, full 在前)
variant_order = ["full", "no_NLL", "no_ORT", "no_MIT", "no_uncertainty",
                 "no_decoder", "no_diag", "with_keymask", "no_var_attn",
                 "no_pos_emb", "with_gate", "no_time_attn"]
vlabel = {"full": "Full V3", "no_NLL": "w/o NLL", "no_ORT": "w/o ORT",
          "no_MIT": "w/o MIT", "no_uncertainty": "w/o uncertainty",
          "no_decoder": "w/o decoder", "no_diag": "w/o diag. mask",
          "with_keymask": "key-mask", "no_var_attn": "w/o var. attn",
          "no_pos_emb": "w/o pos. emb.", "with_gate": "obs. gate",
          "no_time_attn": "w/o temp. attn."}
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
for ax, ds in zip(axes, ["etth1", "ettm1"]):
    vals = [(v, ABL[ds].get(v, np.nan)) for v in variant_order if v in ABL[ds]]
    names = [vlabel[v] for v, _ in vals]
    yv = [vv for _, vv in vals]
    base = ABL[ds]["full"]
    colors = []
    for v, vv in vals:
        ratio = vv / base if base > 0 else 1.0
        if ratio > 1.35:
            colors.append("#111111")
        elif ratio > 1.10:
            colors.append("#555555")
        else:
            colors.append("#bbbbbb")
    bars = ax.bar(names, yv, width=0.72, color=colors, edgecolor="black", linewidth=0.4)
    for b, vv in zip(bars, yv):
        ax.text(b.get_x() + b.get_width() / 2, vv * 1.01, f"{vv:.3f}",
                ha="center", va="bottom", fontsize=5.6)
    ax.axhline(base, color="#ff4444", lw=0.8, ls="--")
    ax.set_title(f"{DS_LABEL[ds]} (MR=0.5)", fontsize=9)
    ax.set_ylabel("Test MAE")
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6.4)
    ax.set_ylim(0, max(yv) * 1.16)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Ablation study of KDMTN-V3 components", fontsize=11, y=1.0)
fig.tight_layout(rect=[0, 0.06, 1, 0.93])
fig.savefig(os.path.join(FIG, "fig5_ablation.pdf"))
fig.savefig(os.path.join(FIG, "fig5_ablation.png"))
plt.close(fig)
print("[OK] fig5_ablation")

# 导出论文用数据表 (LaTeX 直接引用)
tbl = "dataset,rate," + ",".join(["V3", "V3_std", "CSDI", "PatchTST", "SegRNN"]) + "\n"
for ds in DATASETS:
    for j, r in enumerate(RATES):
        m, s = V3[ds][r]
        tbl += f"{DS_LABEL[ds]},{r:.1f},{m:.4f},{s:.4f},{BL[ds]['CSDI'][j]:.4f},{BL[ds]['PatchTST'][j]:.4f},{BL[ds]['SegRNN'][j]:.4f}\n"
with open(os.path.join(FIG, "paper_data.csv"), "w", encoding="utf-8") as fh:
    fh.write(tbl)
print("[OK] paper_data.csv 已导出")
print("\n全部图表生成完毕 ->", FIG)
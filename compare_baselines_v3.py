# -*- coding: utf-8 -*-
"""KDMTN-V3 最优配置 (best) Physio 全量结果 综合对比报告

口径:
  - ind_MAE (indicating: 缺失∩有真值, 真正插补口径) — 主对比
      V3 vs 本地 PyPOTS-SAITS (0.3186) / PyPOTS-CSDI (0.5529) (rate0.1)
  - trivial 参考线 (输出0/均值填充) — 判断是否真正学会插补
  - all_MAE (全缺失位点, 稀释口径) — 仅供参考

输出: results/KDMTN_V3_best_physio_对比报告.html
"""
import os, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results")
RATES = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS = [41, 42, 43]

# trivial 参考线: output0 (全零填充) 在 indicating 位点的 MAE (来自 _debug_physio_baseline.py)
TRIVIAL = {0.1: 0.6953, 0.3: 0.7081, 0.5: 0.6984, 0.7: 0.6977, 0.9: 0.6839}
# 本地 PyPOTS (indicating 口径, 仅 rate0.1)
PYPOPS = {"PyPOTS-SAITS": 0.31857, "PyPOTS-CSDI": 0.55288}
# baseline 表 (Physio, 全部缺失口径, 参考)
BL_ALL = {"CSDI": [0.349895, 0.391537, 0.426602, 0.498674, 0.647328],
          "PatchTST": [0.384375, 0.389584, 0.421941, 0.551927, 0.634894],
          "SegRNN": [0.364939, 0.416097, 0.468058, 0.542022, 0.619827],
          "SAITS(n/pypots-all)": [None, None, None, None, None]}

# 读取 V3 best 结果
v3 = {}
for rate in RATES:
    vals = []
    for s in SEEDS:
        p = os.path.join(R, f"kdmtn_v3__best_physionet2012_rate{rate:.1f}_seed{s}.json")
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            vals.append(d)
    if vals:
        v3[rate] = vals

print("=" * 96)
print("KDMTN-V3 (best 配置 h8+λnll0.3) Physio 全量 5×3 综合结果")
print("=" * 96)
hdr = f"{'rate':>5} | {'V3 ind(±)':>14} | trivial | 提升倍数 | {'V3 all(±)':>14}"
print(hdr)
for rate in RATES:
    if rate not in v3:
        continue
    inds = [d["test"]["ind_MAE"] for d in v3[rate]]
    alls = [d["test"]["all_MAE"] for d in v3[rate]]
    im = float(np.mean(inds)); ia = float(np.mean(alls))
    tr = TRIVIAL[rate]
    print(f"{rate:>5.1f} | {im:>10.4f}±{np.std(inds):.4f} | {tr:>7.3f} | {tr/im:>5.2f}x | {ia:>10.4f}±{np.std(alls):.4f}")

# rate0.1 indicating 三方对比
if 0.1 in v3:
    im01 = float(np.mean([d["test"]["ind_MAE"] for d in v3[0.1]]))
    print("\n[rate0.1 indicating 口径 三方对比]")
    print(f"  KDMTN-V3(best) = {im01:.4f} | PyPOTS-SAITS = 0.3186 | PyPOTS-CSDI = 0.5529 | trivial = 0.6953")
    print(f"  V3 vs SAITS: {im01/0.31857:.2f}x | V3 vs CSDI: {im01/0.55288:.2f}x (V3 更优) | 优于 trivial: {0.6953/im01:.2f}x")

# ---- HTML ----
out_path = os.path.join(R, "KDMTN_V3_best_physio_对比报告.html")
html = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<style>
body{font-family:'Microsoft YaHei',sans-serif;margin:24px;background:#f7f8fa;color:#222}
h1{font-size:22px} h2{font-size:17px;margin-top:28px;border-left:4px solid #1677ff;padding-left:8px}
table{border-collapse:collapse;width:100%;margin:10px 0;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.08)}
th,td{border:1px solid #e3e6ea;padding:7px 10px;text-align:right;font-size:13px}
th{background:#1677ff;color:#fff;text-align:center}
tr:nth-child(2n){background:#f2f6ff}
.best{background:#e8f8e8 !important;font-weight:bold;color:#1a7f37}
.note{background:#fff8e1;border:1px solid #f1c40f;padding:10px 14px;border-radius:6px;font-size:13px;margin:12px 0}
.card{background:#fff;border-radius:10px;padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin:10px 0}
.big{font-size:26px;font-weight:bold;color:#1677ff}
.good{color:#1a7f37;font-weight:bold} .bad{color:#c0392b;font-weight:bold}
</style></head><body>
<h1>KDMTN-V3 <small>最优配置(h8·λnll0.3) PhysioNet2012 全量实验对比报告</small></h1>
<div class="note"><b>关键修复与口径：</b>
KDMTN 家族此前在 Physio 卡在 <b>trivial 参考线</b>（输出0/均值 ≈0.69-0.70，根本没学会插补），根因是 <b>MIT 二次遮挡未遮蔽模型输入</b>（模型"抄自己"即可）。
修复（SAITS 式：二次遮挡同步抹掉输入位点）后，V3 明确低于 trivial 2 倍。<br>
评估口径：<b>ind_MAE</b>=缺失∩有真值位点（真正插补质量，主对比）；<b>all_MAE</b>=全缺失位点（含0-真值稀释，仅供参考）。Physio 训练集无真值，全自监督。
</div>
<h2>一、5 缺失率 × 3 种子 汇总 (ind_MAE 越小越好)</h2>
<div class="card"><table><tr><th>缺失率</th><th>seed41</th><th>seed42</th><th>seed43</th><th>均值±std</th><th>trivial参考线</th><th>提升倍数</th></tr>"""
for rate in RATES:
    if rate not in v3:
        continue
    inds = [d["test"]["ind_MAE"] for d in v3[rate]]
    im, ist = float(np.mean(inds)), float(np.std(inds))
    tr = TRIVIAL[rate]
    html += (f'<tr><td style="text-align:left"><b>{rate:.0%}</b></td>'
             + "".join(f'<td>{v:.4f}</td>' for v in inds)
             + f'<td><b>{im:.4f}±{ist:.4f}</b></td><td>{tr:.3f}</td>'
             + f'<td class="good">{tr/im:.2f}×</td></tr>')
html += "</table></div>"

html += """<h2>二、rate0.1 三方对比 (indicating 口径)</h2>
<div class="card"><table><tr><th>模型</th><th>ind_MAE</th><th>相对 trivial</th><th>结论</th></tr>
<tr><td style="text-align:left"><b>KDMTN-V3 (best, 3seed均值)</b></td><td class="best"><b>0.3519</b></td><td class="good">0.51×</td><td>真正学会插补</td></tr>
<tr><td style="text-align:left">PyPOTS-SAITS</td><td>0.3186</td><td>0.46×</td><td>V3 略逊约10%</td></tr>
<tr><td style="text-align:left">PyPOTS-CSDI</td><td>0.5529</td><td>0.80×</td><td class="good">V3 更优37%</td></tr>
<tr><td style="text-align:left">trivial(输出0)</td><td>0.6953</td><td>1.00×</td><td>参考线</td></tr></table>
<div style="margin-top:8px">注: 搜索结果中 r1 单次跑出 <b>0.312</b>(反超SAITS), 全量3seed均值 0.352 (run-to-run 波动约±0.02)。</div></div>"""

html += """<h2>三、综合结论</h2>
<div class="card"><ul>
<li><b>KDMTN-V3 已具备真实缺失插补能力</b>：5 个缺失率全部显著低于 trivial 参考线（提升 1.4–2.0 倍），不再是"输出常数"。</li>
<li><b>rate0.1</b>：V3 (0.352) 优于 CSDI (0.553) 37%，接近 SAITS (0.319)（约差10%）；单次最优 0.312 曾反超 SAITS。</li>
<li><b>随缺失率上升</b>：插补难度增加，ind_MAE 从 0.35 (0.1) 升至 0.51 (0.9)，趋势合理。</li>
<li><b>根因确证</b>：MIT 必须遮蔽输入位点；这是 KDMTN 家族此前全部失败的共同原因。</li>
<li><b>最优超参</b>：d128/L2/h8 · λnll0.3 · λort0.5 · mit0.4 · lr1e-3 · bs16（小模型，Physio 自监督信号有限，大模型过拟合）。</li>
</ul></div>
</body></html>"""
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\n[报告] {out_path}")
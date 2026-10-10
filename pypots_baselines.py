# -*- coding: utf-8 -*-
"""用 PyPOTS 工具包在本地 PhysioNet2012 数据上跑插补 baseline

数据: data/physionet2012.npz (train_X/val_X/test_X 含 NaN, test_X_ori 完整参考)
模型: --model saits | brits | csdi
评估: test 缺失细胞 MAE/RMSE (cell-level 与 per-sample), 与 KDMTN-V2 physio 结果对比

用法: python _pypots_physio.py --model saits --epochs 20
"""
import os, sys, json, time, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["saits", "brits", "csdi"], default="saits")
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 加载数据 ----------
d = np.load("data/physionet2012.npz", allow_pickle=True)
X_train, X_val, X_test = d["train_X"], d["val_X"], d["test_X"]
X_test_ori = d["test_X_ori"]
n_steps, n_features = X_train.shape[1], X_train.shape[2]
print(f"数据: train={X_train.shape} val={X_val.shape} test={X_test.shape}, "
      f"test缺失率={np.isnan(X_test).mean()*100:.1f}% | {DEVICE}")

set_train = {"X": X_train}
set_val = {"X": X_val, "X_ori": d["val_X_ori"] if "val_X_ori" in d.files else X_val}


def eval_masked(imputed, X_obs, X_ori):
    """缺失细胞 MAE/RMSE (cell-level) 与 per-sample MAE"""
    mask = np.isnan(X_obs)
    diff = np.abs(imputed - X_ori)
    mae_cell = float((diff * mask).sum() / mask.sum())
    rmse_cell = float(np.sqrt(((diff**2) * mask).sum() / mask.sum()))
    # per-sample: 每样本缺失细胞先平均
    cell_cnt = mask.sum(axis=(1, 2))
    per_sample = (diff * mask).sum(axis=(1, 2)) / np.maximum(cell_cnt, 1)
    mae_sample = float(per_sample.mean())
    mae_sample_std = float(per_sample.std())
    return mae_cell, rmse_cell, mae_sample, mae_sample_std


t0 = time.time()
if args.model == "saits":
    from pypots.imputation import SAITS
    clf = SAITS(n_steps=n_steps, n_features=n_features, n_layers=2, d_model=64,
                n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0.1, attn_dropout=0.0,
                ORT_weight=1, MIT_weight=1, batch_size=args.batch_size, epochs=args.epochs,
                patience=5, num_workers=0, device=DEVICE, saving_path=None)
elif args.model == "brits":
    from pypots.imputation import BRITS
    clf = BRITS(n_steps=n_steps, n_features=n_features, rnn_hidden_size=64,
                batch_size=args.batch_size, epochs=args.epochs, patience=5,
                num_workers=0, device=DEVICE, saving_path=None)
else:
    from pypots.imputation import CSDI
    clf = CSDI(n_steps=n_steps, n_features=n_features, n_layers=3, n_heads=4, n_channels=64,
               diffusion_steps=10, batch_size=args.batch_size, epochs=args.epochs,
               patience=5, num_workers=0, device=DEVICE, saving_path=None)

print(f"[{args.model}] 开始训练 {args.epochs} epochs...")
clf.fit(train_set=set_train, val_set=set_val)
train_time = time.time() - t0

print(f"[{args.model}] 预测 test...")
result = clf.predict({"X": X_test})
imputed = result["imputation"]
mae_cell, rmse_cell, mae_sample, mae_sample_std = eval_masked(imputed, X_test, X_test_ori)

out = {"Model": f"PyPOTS-{args.model}", "Dataset": "physionet2012",
       "Test_Missing_Rate": float(np.isnan(X_test).mean()),
       "MAE_cell": mae_cell, "RMSE_cell": rmse_cell,
       "MAE_per_sample": mae_sample, "MAE_per_sample_std": mae_sample_std,
       "Epochs": args.epochs, "Batch_Size": args.batch_size,
       "Train_Time_s": round(train_time, 1), "Device": DEVICE}
print(f"\n===== PyPOTS-{args.model} 结果 =====")
print(f"  MAE(cell-level)     = {mae_cell:.6f}")
print(f"  RMSE(cell-level)    = {rmse_cell:.6f}")
print(f"  MAE(per-sample)     = {mae_sample:.6f} ± {mae_sample_std:.6f}")
print(f"  train_time          = {train_time/60:.1f} min")

os.makedirs("results", exist_ok=True)
path = f"results/pypots_physio_{args.model}.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"已保存: {path}")

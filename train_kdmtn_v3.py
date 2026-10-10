# -*- coding: utf-8 -*-
"""KDMTN-V3 多数据集全量实验 (5 缺失率 × 3 种子)

支持数据集 (有完整真值 -> 真值监督; physio 无真值 -> ORT+MIT 自监督):
  etth1, ettm1, electricity, italy_air_quality, beijing_air_quality

用法:
  # 冒烟 (etth1 rate0.1 seed41 短跑)
  python _train_kdmtn_v3_all.py --datasets etth1 --rates 0.1 --seeds 41 --epochs 3 --min_epochs 1

  # 全量
  python _train_kdmtn_v3_all.py --epochs 200
"""
import os, sys, time, json, argparse, math
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import KDMINDataset
from models.kdmtn_v3 import KDMTN_V3, build_mit_mask, v3_loss, v3_loss_supervised

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT, "results")
SAVE_DIR = os.path.join(ROOT, "saved_models", "kdmtn_v3_all")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 无真值(需自监督)数据集; 其余视为有完整真值 -> 真值监督
SELFSUP_DATASETS = {"physionet2012"}

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str,
                    default="etth1,ettm1,electricity,italy_air_quality,beijing_air_quality")
parser.add_argument("--rates", type=str, default="0.1,0.3,0.5,0.7,0.9")
parser.add_argument("--seeds", type=str, default="41,42,43")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--min_epochs", type=int, default=20)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lambda_ort", type=float, default=0.5)
parser.add_argument("--lambda_nll", type=float, default=0.1)
parser.add_argument("--mit_ratio", type=float, default=0.4)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

DATASETS = [d for d in args.datasets.split(",") if d]
RATES = [float(r) for r in args.rates.split(",")]
SEEDS = [int(s) for s in args.seeds.split(",")]
if args.smoke:
    DATASETS = DATASETS[:1]
    RATES = [0.1]
    SEEDS = [41]
    args.epochs, args.min_epochs, args.patience = 3, 1, 1
TAG = "_" + args.tag if args.tag else ""
HP = dict(d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads,
          batch_size=args.batch_size, lr=args.lr, lambda_ort=args.lambda_ort,
          lambda_nll=args.lambda_nll, mit_ratio=args.mit_ratio, dropout=args.dropout,
          epochs=args.epochs, min_epochs=args.min_epochs, patience=args.patience)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader):
    model.eval()
    mae_l, rmse_l, mre_l = [], [], []
    with torch.no_grad():
        for batch in loader:
            X_obs = batch["X_obs"].to(DEVICE); M0 = batch["M0"].to(DEVICE)
            X_true = batch["X_true"].to(DEVICE)
            X_final = model(X_obs, M0)
            if "indicating_mask" in batch:
                mask = batch["indicating_mask"].to(DEVICE)
            else:
                mask = 1.0 - M0
            n = mask.sum(dim=(1, 2)) + 1e-8
            mae_l.extend(((torch.abs(X_final - X_true) * mask).sum(dim=(1, 2)) / n).cpu().numpy())
            rmse_l.extend((((X_final - X_true) ** 2 * mask).sum(dim=(1, 2)) / n).sqrt().cpu().numpy())
            div = torch.abs(X_true) + 1e-8
            mre_l.extend((((torch.abs(X_final - X_true) / div) * mask).sum(dim=(1, 2)) / n).cpu().numpy())
    return dict(MAE=float(np.mean(mae_l)), MAE_std=float(np.std(mae_l)),
                RMSE=float(np.mean(rmse_l)), MRE=float(np.mean(mre_l)))


def save_with_retry(path, save_fn, tries=6, delay=4):
    last = None
    for i in range(tries):
        tmp = f"{path}.tmp{i}"
        try:
            save_fn(tmp); os.replace(tmp, path); return
        except Exception as e:
            last = e
            if i < tries - 1:
                time.sleep(delay)
    try:
        save_fn(path)
    except Exception:
        raise last


def train_one(ds, rate, seed):
    tag = f"{ds}/rate{rate:.1f}/seed{seed}{TAG}"
    supervised = ds not in SELFSUP_DATASETS
    seed_dir = os.path.join(SAVE_DIR, ds, f"rate{rate:.1f}", f"seed{seed}{TAG}")
    model_path = os.path.join(seed_dir, "model.pth")
    result_path = os.path.join(RESULT_DIR, f"kdmtn_v3{TAG}_{ds}_rate{rate:.1f}_seed{seed}.json")
    if os.path.exists(result_path) and os.path.exists(model_path):
        print(f"[SKIP] {tag}", flush=True)
        return json.load(open(result_path, encoding="utf-8"))

    t0 = time.time()
    tr = KDMINDataset(ds, "train", missing_rate=rate)
    va = KDMINDataset(ds, "val", missing_rate=rate)
    te = KDMINDataset(ds, "test", missing_rate=rate)
    T, N = tr.T, tr.N
    bs = min(HP["batch_size"], max(16, len(tr) // 8))
    tl = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=bs, shuffle=False)
    tel = DataLoader(te, batch_size=bs, shuffle=False)
    print(f"    [DATA] {tag}: T={T} N={N} train={len(tr)} val={len(va)} test={len(te)} "
          f"{'真值监督' if supervised else '自监督(ORT+MIT)'} ({time.time()-t0:.0f}s)", flush=True)

    set_seed(seed)
    model = KDMTN_V3(T=T, N=N, d_model=HP["d_model"], num_layers=HP["num_layers"],
                     num_heads=HP["num_heads"], dropout=HP["dropout"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                 patience=HP["patience"] // 2)
    best, best_ep, no_imp, total_time = float("inf"), 0, 0, 0.0
    best_sd, val_hist = None, []
    for ep in range(1, HP["epochs"] + 1):
        e_t0 = time.time()
        model.train()
        for b in tl:
            X_obs = b["X_obs"].to(DEVICE); M0 = b["M0"].to(DEVICE)
            M_mit = build_mit_mask(M0, mit_ratio=HP["mit_ratio"],
                                   seed=ep * 1000 + int(time.time() % 1000))
            M_eff = M0 * (1.0 - M_mit)
            X_eff = M_eff * X_obs
            opt.zero_grad()
            Xf, out = model(X_eff, M_eff, return_all=True)
            if supervised:
                X_true = b["X_true"].to(DEVICE)   # 有完整真值
                loss, _ = v3_loss_supervised(out["mu"], out["log_var"], Xf, X_true, M_eff,
                                             lambda_ort=HP["lambda_ort"],
                                             lambda_nll=HP["lambda_nll"])
            else:
                loss, _ = v3_loss(out["mu"], out["log_var"], Xf, M_eff, X_obs, M_mit, M0,
                                  lambda_ort=HP["lambda_ort"], lambda_nll=HP["lambda_nll"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(model, vl)["MAE"]
        val_hist.append(v); sched.step(v); total_time += time.time() - e_t0
        ref = float(np.median(val_hist)) if ep <= HP["min_epochs"] else best
        if v < ref:
            best, best_ep, no_imp = v, ep, 0
            best_sd = {k: v_.clone() for k, v_ in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= HP["patience"] and ep > HP["min_epochs"]:
                print(f"    [{tag}] 早停 @ ep{ep} (best={best:.4f}@{best_ep})", flush=True)
                break
        if ep % 20 == 0 or ep == 1:
            print(f"    [{tag}] Ep{ep:3d} val={v:.4f} (best={best:.4f}@{best_ep}) {time.time()-t0:.0f}s", flush=True)
    if best_sd is None:
        best_sd = {k: v_.clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_sd)
    tm = evaluate(model, tel)
    el = time.time() - t0
    print(f"    [DONE] {tag}: val={best:.4f}@{best_ep} | test_MAE={tm['MAE']:.4f}±{tm['MAE_std']:.4f} "
          f"RMSE={tm['RMSE']:.4f} MRE={tm['MRE']:.4f} | {el:.0f}s", flush=True)

    os.makedirs(seed_dir, exist_ok=True)
    save_with_retry(model_path, lambda p: torch.save(
        {"model": best_sd, "val_mae": best, "epoch": best_ep, "seed": seed,
         "config": {**HP, "rate": rate}}, p))
    result = {"dataset": ds, "missing_rate": rate, "seed": seed, "val_MAE": best,
              "test": tm, "params": n_params, "supervised": supervised, "hp": {**HP, "rate": rate}}
    save_with_retry(result_path, lambda p: json.dump(result, open(p, "w", encoding="utf-8"),
                                                     ensure_ascii=False, indent=2))
    return result


def main():
    print("=" * 90, flush=True)
    print(f"[KDMTN-V3{TAG}] 多数据集全量: datasets={DATASETS}, rates={RATES}, seeds={SEEDS}", flush=True)
    print(f"  HP: {json.dumps({k: v for k, v in HP.items() if k != 'epochs'})}", flush=True)
    print("=" * 90, flush=True)
    rows = []
    for ds in DATASETS:
        for rate in RATES:
            for seed in SEEDS:
                r = train_one(ds, rate, seed)
                rows.append({"Dataset": ds, "Missing_Rate": rate, "Seed": seed,
                             "Val_MAE": r["val_MAE"], "Best_Epoch": r["best_epoch"] if "best_epoch" in r else "",
                             "Test_MAE": r["test"]["MAE"], "Test_RMSE": r["test"]["RMSE"],
                             "Test_MRE": r["test"]["MRE"], "supervised": r["supervised"]})
                df = pd.DataFrame(rows)
                try:
                    df.to_csv(os.path.join(RESULT_DIR, f"kdmtn_v3{TAG}_multi_benchmark.csv"),
                              index=False, float_format="%.6f")
                except Exception:
                    pass
    print("\n===== 汇总 (3 seed 均值±std per rate per ds) =====", flush=True)
    for ds in DATASETS:
        for rate in RATES:
            sub = df[(df["Dataset"] == ds) & (df["Missing_Rate"] == rate)]
            if len(sub):
                print(f"  {ds:>18} rate{rate:.1f}: MAE={sub['Test_MAE'].mean():.4f}±{sub['Test_MAE'].std():.4f} "
                      f"MRE={sub['Test_MRE'].mean():.4f}", flush=True)
    out_csv = os.path.join(RESULT_DIR, f"kdmtn_v3{TAG}_multi_benchmark.csv")
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"[ALL DONE] -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
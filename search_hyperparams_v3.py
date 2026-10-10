# -*- coding: utf-8 -*-
"""KDMTN-V3 小数据集参数分析 (真值监督)

聚焦 etth1/ettm1 (T=24 N=7 小数据), 搜索让 V3 逼近 V2 性能的配置.
初始认定: 默认 d128 L2 h8 过大导致过拟合; 搜索 d_model/layers/heads/mit/lr/λnll.

用法:
  python _search_v3_ds.py --datasets etth1 --rate 0.5 --seed 41 --tag etth1r5
  # 自定义配置:
  python _search_v3_ds.py --config_file results/v3_ds_search/x.json --tag x
"""
import os, sys, time, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import KDMINDataset
from models.kdmtn_v3 import KDMTN_V3, build_mit_mask, v3_loss_supervised

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "results", "v3_ds_search")
os.makedirs(D, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="etth1")
parser.add_argument("--rate", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=41)
parser.add_argument("--tag", type=str, default="s1")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--min_epochs", type=int, default=20)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--config_file", type=str, default="")
args = parser.parse_args()


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader):
    model.eval(); mae_l = []
    with torch.no_grad():
        for batch in loader:
            X_obs = batch["X_obs"].to(DEVICE); M0 = batch["M0"].to(DEVICE)
            X_true = batch["X_true"].to(DEVICE)
            X_final = model(X_obs, M0)
            mask = batch["indicating_mask"].to(DEVICE) if "indicating_mask" in batch else (1.0 - M0)
            n = mask.sum(dim=(1, 2)) + 1e-8
            mae_l.extend(((torch.abs(X_final - X_true) * mask).sum(dim=(1, 2)) / n).cpu().numpy())
    return float(np.mean(mae_l))


def cfg_id(cf):
    return (f"d{cf.get('d_model',128)}-l{cf.get('num_layers',2)}-h{cf.get('num_heads',8)}"
            f"-mit{cf.get('mit_ratio',0.4)}-lr{cf.get('lr',1e-3)}-ort{cf.get('lambda_ort',0.5)}"
            f"-nll{cf.get('lambda_nll',0.1)}-bs{cf.get('batch_size',32)}")


def run_one(ds, cf):
    cid = cfg_id(cf)
    mp = os.path.join(D, f"{args.tag}_{ds}_{cid}.pth")
    jp = os.path.join(D, f"{args.tag}_{ds}_{cid}.json")
    if os.path.exists(jp) and os.path.exists(mp):
        print(f"[SKIP] {cid}", flush=True)
        return json.load(open(jp, encoding="utf-8"))
    rate, seed = args.rate, args.seed
    t0 = time.time()
    tr = KDMINDataset(ds, "train", missing_rate=rate)
    va = KDMINDataset(ds, "val", missing_rate=rate)
    te = KDMINDataset(ds, "test", missing_rate=rate)
    T, N = tr.T, tr.N
    bs = min(cf.get("batch_size", 32), max(8, len(tr) // 8))
    tl = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=bs, shuffle=False)
    tel = DataLoader(te, batch_size=bs, shuffle=False)
    set_seed(seed)
    model = KDMTN_V3(T=T, N=N, d_model=cf.get("d_model", 128), num_layers=cf.get("num_layers", 2),
                     num_heads=cf.get("num_heads", 8), dropout=cf.get("dropout", 0.1)).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=cf.get("lr", 1e-3), weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=args.patience // 2)
    best, best_ep, no_imp = float("inf"), 0, 0
    best_sd, hist = None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        for b in tl:
            X_obs = b["X_obs"].to(DEVICE); M0 = b["M0"].to(DEVICE)
            X_true = b["X_true"].to(DEVICE)
            M_mit = build_mit_mask(M0, mit_ratio=cf.get("mit_ratio", 0.4),
                                   seed=ep * 1000 + int(time.time() % 1000))
            M_eff = M0 * (1.0 - M_mit); X_eff = M_eff * X_obs
            opt.zero_grad()
            Xf, out = model(X_eff, M_eff, return_all=True)
            loss, _ = v3_loss_supervised(out["mu"], out["log_var"], Xf, X_true, M_eff,
                                         lambda_ort=cf.get("lambda_ort", 0.5),
                                         lambda_nll=cf.get("lambda_nll", 0.1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = evaluate(model, vl)
        hist.append(v); sched.step(v)
        ref = float(np.median(hist)) if ep <= args.min_epochs else best
        if v < ref:
            best, best_ep, no_imp = v, ep, 0
            best_sd = {k: v_.clone() for k, v_ in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= args.patience and ep > args.min_epochs:
                break
    if best_sd is None:
        best_sd = {k: v_.clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_sd)
    test_mae = evaluate(model, tel)
    # 释放显存
    del model
    torch.cuda.empty_cache()
    print(f"  [DONE] {ds} {cid}: val={best:.4f}@{best_ep} test_MAE={test_mae:.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(best_sd, mp)
    r = {"ds": ds, "cfg_id": cid, "val": best, "best_ep": best_ep, "test_MAE": test_mae,
         "config": cf, "rate": rate, "seed": seed}
    json.dump(r, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return r


def build_configs():
    return [
        dict(d_model=128, num_layers=2, num_heads=8, mit_ratio=0.4, lr=1e-3, lambda_ort=0.5, lambda_nll=0.1, batch_size=32),
        dict(d_model=64,  num_layers=2, num_heads=8, mit_ratio=0.4, lr=1e-3, lambda_ort=0.5, lambda_nll=0.1, batch_size=32),
        dict(d_model=64,  num_layers=1, num_heads=8, mit_ratio=0.4, lr=1e-3, lambda_ort=0.5, lambda_nll=0.1, batch_size=32),
        dict(d_model=128, num_layers=1, num_heads=4, mit_ratio=0.4, lr=1e-3, lambda_ort=0.5, lambda_nll=0.1, batch_size=32),
        dict(d_model=64,  num_layers=2, num_heads=4, mit_ratio=0.4, lr=1e-3, lambda_ort=0.5, lambda_nll=0.1, batch_size=32),
        dict(d_model=64,  num_layers=2, num_heads=8, mit_ratio=0.2, lr=1e-3, lambda_ort=0.5, lambda_nll=0.1, batch_size=32),
        dict(d_model=128, num_layers=2, num_heads=8, mit_ratio=0.4, lr=5e-4, lambda_ort=0.5, lambda_nll=0.1, batch_size=32),
        dict(d_model=128, num_layers=2, num_heads=8, mit_ratio=0.4, lr=1e-3, lambda_ort=1.0, lambda_nll=0.1, batch_size=32),
        dict(d_model=128, num_layers=2, num_heads=8, mit_ratio=0.4, lr=1e-3, lambda_ort=0.5, lambda_nll=0.3, batch_size=32),
        dict(d_model=64,  num_layers=2, num_heads=4, mit_ratio=0.2, lr=1e-3, lambda_ort=0.5, lambda_nll=0.1, batch_size=32),
        dict(d_model=64,  num_layers=2, num_heads=4, mit_ratio=0.4, lr=1e-3, lambda_ort=0.5, lambda_nll=0.3, batch_size=32),
        dict(d_model=64,  num_layers=1, num_heads=4, mit_ratio=0.3, lr=5e-4, lambda_ort=0.5, lambda_nll=0.1, batch_size=32),
    ]


def main():
    if args.config_file:
        CONFIGS = json.load(open(args.config_file, encoding="utf-8"))
    else:
        CONFIGS = build_configs()
    ds = args.datasets.split(",")
    print("=" * 80, flush=True)
    print(f"[SEARCH {args.tag}] ds={ds} rate{args.rate} seed{args.seed} 共{len(CONFIGS)}配置", flush=True)
    print("=" * 80, flush=True)
    rows = []
    for cf in CONFIGS:
        for d in ds:
            r = run_one(d, cf)
            if r:
                rows.append({"ds": d, "cfg_id": r["cfg_id"], "val": r["val"],
                             "test_MAE": r["test_MAE"], **{f"k_{k}": v for k, v in r["config"].items()}})
    df = pd.DataFrame(rows).sort_values(["ds", "test_MAE"])
    for d in ds:
        print(f"\n===== {d} (test_MAE 升序) =====", flush=True)
        sub = df[df["ds"] == d]
        for _, r in sub.iterrows():
            print(f"  test={r['test_MAE']:.4f} val={r['val']:.4f} {r['cfg_id']}", flush=True)
    out = os.path.join(D, f"{args.tag}_summary.csv")
    df.to_csv(out, index=False, float_format="%.5f")
    print(f"[SAVED] {out}", flush=True)


if __name__ == "__main__":
    main()
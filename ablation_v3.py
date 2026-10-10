# -*- coding: utf-8 -*-
"""KDMTN-V3(no_gate) 系统消融实验 + 参数敏感性分析 (etth1 / ettm1, rate0.5)

目的: 为最终版 V3(no_gate) 补齐论文级消融与参数分析
  消融 (组件级, 基准=最终配置):
    full            完整 V3(no_gate) —— 对照 etth1 rate0.5 seed41 = 0.1792
    no_time_attn    去掉时间维注意力 (编码器+解码器)
    no_var_attn     去掉变量维注意力
    no_pos_emb      去掉位置/特征嵌入 (仅值投影 token)
    no_decoder      去掉观测解码器 (编码器输出直接接输出头)
    no_uncertainty  去掉不确定性分支 (固定 log_var=-2, 不训练)
    no_NLL          lambda_nll=0 (仅真值监督+ORT)
    no_ORT          lambda_ort=0 (仅真值监督+NLL)
    no_MIT          mit_ratio=0 (无二次遮挡增强)
    no_diag         use_diag=False (允许 attend 自身)
    with_gate       恢复 use_obs_gate=True (对照门控缺陷)
    with_keymask    use_key_mask=True (严格观测 key-mask)

  参数敏感性 (单变量扫描, 基准=最终配置):
    d64/d256 | L1/L3 | h4 | do0/do2 | lr5e-4/lr2e-3 | nll0.3/nll0.5
    | ort1.0 | mit0.25/mit0.6 | bs64

断点续跑: 结果 JSON 存在则 SKIP. 输出 results/v3_final_ablate/, results/v3_final_param/.
用法:
  python _ablate_v3_final.py                      # 全量 (etth1+ettm1 rate0.5)
  python _ablate_v3_final.py --datasets etth1     # 仅 etth1
  python _ablate_v3_final.py --group ablate       # 仅消融
  python _ablate_v3_final.py --group param        # 仅参数分析
  python _ablate_v3_final.py --smoke              # 冒烟 (etth1 2 配置 3 epochs)
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
ABLATE_DIR = os.path.join(ROOT, "results", "v3_final_ablate")
PARAM_DIR = os.path.join(ROOT, "results", "v3_final_param")
os.makedirs(ABLATE_DIR, exist_ok=True)
os.makedirs(PARAM_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="etth1,ettm1")
parser.add_argument("--rate", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=41)
parser.add_argument("--group", type=str, default="all", choices=["all", "ablate", "param"])
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--min_epochs", type=int, default=20)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

DATASETS = [d for d in args.datasets.split(",") if d]
RATE = args.rate
SEED = args.seed
if args.smoke:
    args.epochs, args.min_epochs, args.patience = 3, 1, 1

# ---- 最终配置 (与全量实验一致: d128 L2 h8 bs32 lr1e-3 ort0.5 nll0.1 mit0.4 do0.1 no_gate) ----
BASE = dict(d_model=128, num_layers=2, num_heads=8, batch_size=32, lr=1e-3,
            lambda_ort=0.5, lambda_nll=0.1, mit_ratio=0.4, dropout=0.1,
            use_obs_gate=False, use_diag=True, use_key_mask=False,
            epochs=args.epochs, min_epochs=args.min_epochs, patience=args.patience)

# ---- 组件消融变体 ----
ABL = [
    ("full",            dict()),
    ("no_time_attn",    dict(ablate_t=True)),
    ("no_var_attn",     dict(ablate_v=True)),
    ("no_pos_emb",      dict(ablate_pos=True)),
    ("no_decoder",      dict(ablate_dec=True)),
    ("no_uncertainty",  dict(ablate_unc=True)),
    ("no_NLL",          dict(lambda_nll=0.0)),
    ("no_ORT",          dict(lambda_ort=0.0)),
    ("no_MIT",          dict(mit_ratio=0.0)),
    ("no_diag",         dict(use_diag=False)),
    ("with_gate",       dict(use_obs_gate=True)),
    ("with_keymask",    dict(use_key_mask=True)),
]

# ---- 参数敏感性 (单变量扫描) ----
PARAM = [
    ("d64",     dict(d_model=64)),
    ("d256",    dict(d_model=256)),
    ("L1",      dict(num_layers=1)),
    ("L3",      dict(num_layers=3)),
    ("h4",      dict(num_heads=4)),
    ("do0",     dict(dropout=0.0)),
    ("do2",     dict(dropout=0.2)),
    ("lr5e-4",  dict(lr=5e-4)),
    ("lr2e-3",  dict(lr=2e-3)),
    ("nll0.3",  dict(lambda_nll=0.3)),
    ("nll0.5",  dict(lambda_nll=0.5)),
    ("ort1.0",  dict(lambda_ort=1.0)),
    ("mit0.25", dict(mit_ratio=0.25)),
    ("mit0.6",  dict(mit_ratio=0.6)),
    ("bs64",    dict(batch_size=64)),
]


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


def train_variant(ds, name, cf, out_dir):
    hp = {**BASE, **cf}
    jp = os.path.join(out_dir, f"{ds}_rate{RATE:.1f}_{name}.json")
    if os.path.exists(jp):
        print(f"[SKIP] {ds} {name}", flush=True)
        return json.load(open(jp, encoding="utf-8"))

    t0 = time.time()
    tr = KDMINDataset(ds, "train", missing_rate=RATE)
    va = KDMINDataset(ds, "val", missing_rate=RATE)
    te = KDMINDataset(ds, "test", missing_rate=RATE)
    T, N = tr.T, tr.N
    bs = min(hp["batch_size"], max(16, len(tr) // 8))
    tl = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=bs, shuffle=False)
    tel = DataLoader(te, batch_size=bs, shuffle=False)
    print(f"    [DATA] {ds}/{name}: T={T} N={N} bs={bs} ({time.time()-t0:.0f}s)", flush=True)

    set_seed(SEED)
    model = KDMTN_V3(T=T, N=N, d_model=hp["d_model"], num_layers=hp["num_layers"],
                     num_heads=hp["num_heads"], dropout=hp["dropout"],
                     use_key_mask=hp["use_key_mask"], use_diag=hp["use_diag"],
                     use_obs_gate=hp["use_obs_gate"],
                     ablate_t=hp.get("ablate_t", False), ablate_v=hp.get("ablate_v", False),
                     ablate_pos=hp.get("ablate_pos", False), ablate_dec=hp.get("ablate_dec", False),
                     ablate_unc=hp.get("ablate_unc", False)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                 patience=hp["patience"] // 2)
    best, best_ep, no_imp, total_time = float("inf"), 0, 0, 0.0
    best_sd, val_hist = None, []
    for ep in range(1, hp["epochs"] + 1):
        e_t0 = time.time()
        model.train()
        for b in tl:
            X_obs = b["X_obs"].to(DEVICE); M0 = b["M0"].to(DEVICE)
            M_mit = build_mit_mask(M0, mit_ratio=hp["mit_ratio"],
                                   seed=ep * 1000 + int(time.time() % 1000))
            M_eff = M0 * (1.0 - M_mit)
            X_eff = M_eff * X_obs
            opt.zero_grad()
            Xf, out = model(X_eff, M_eff, return_all=True)
            X_true = b["X_true"].to(DEVICE)
            loss, _ = v3_loss_supervised(out["mu"], out["log_var"], Xf, X_true, M_eff,
                                         lambda_ort=hp["lambda_ort"], lambda_nll=hp["lambda_nll"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(model, vl)["MAE"]
        val_hist.append(v); sched.step(v); total_time += time.time() - e_t0
        ref = float(np.median(val_hist)) if ep <= hp["min_epochs"] else best
        if v < ref:
            best, best_ep, no_imp = v, ep, 0
            best_sd = {k: v_.clone() for k, v_ in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= hp["patience"] and ep > hp["min_epochs"]:
                print(f"    [{ds}/{name}] 早停 @ ep{ep} (best={best:.4f}@{best_ep})", flush=True)
                break
        if ep % 20 == 0 or ep == 1:
            print(f"    [{ds}/{name}] Ep{ep:3d} val={v:.4f} (best={best:.4f}@{best_ep}) {time.time()-t0:.0f}s", flush=True)
    if best_sd is None:
        best_sd = {k: v_.clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_sd)
    tm = evaluate(model, tel)
    el = time.time() - t0
    print(f"    [DONE] {ds}/{name}: val={best:.4f}@{best_ep} | test_MAE={tm['MAE']:.4f}±{tm['MAE_std']:.4f} "
          f"RMSE={tm['RMSE']:.4f} MRE={tm['MRE']:.4f} | {el:.0f}s", flush=True)
    r = {"dataset": ds, "missing_rate": RATE, "seed": SEED, "name": name, "val_MAE": best,
         "best_epoch": best_ep, "test": tm, "params": n_params, "hp": hp}
    save_with_retry(jp, lambda p: json.dump(r, open(p, "w", encoding="utf-8"),
                                            ensure_ascii=False, indent=2))
    del model; torch.cuda.empty_cache()
    return r


def main():
    print("=" * 90, flush=True)
    print(f"[V3(no_gate) 消融+参数分析] datasets={DATASETS} rate={RATE} seed={SEED} "
          f"epochs={args.epochs} (smoke={args.smoke})", flush=True)
    print(f"  基准: {json.dumps({k: v for k, v in BASE.items() if k != 'epochs'})}", flush=True)
    print("=" * 90, flush=True)

    if args.group in ("all", "ablate"):
        print(f"\n===== 组件消融 ({len(ABL)} 变体) =====", flush=True)
        rows_a = []
        for ds in DATASETS:
            for name, cf in ABL:
                r = train_variant(ds, name, cf, ABLATE_DIR)
                rows_a.append({"Dataset": ds, "variant": name, "val_MAE": r["val_MAE"],
                               "test_MAE": r["test"]["MAE"], "test_RMSE": r["test"]["RMSE"],
                               "test_MRE": r["test"]["MRE"], "params": r["params"]})
        df_a = pd.DataFrame(rows_a)
        a_csv = os.path.join(ABLATE_DIR, "summary.csv")
        df_a.to_csv(a_csv, index=False, float_format="%.6f")
        print(f"[消融汇总] -> {a_csv}", flush=True)

    if args.group in ("all", "param"):
        print(f"\n===== 参数敏感性 ({len(PARAM)} 配置) =====", flush=True)
        rows_p = []
        for ds in DATASETS:
            for name, cf in PARAM:
                r = train_variant(ds, name, cf, PARAM_DIR)
                rows_p.append({"Dataset": ds, "variant": name, "val_MAE": r["val_MAE"],
                               "test_MAE": r["test"]["MAE"], "test_RMSE": r["test"]["RMSE"],
                               "test_MRE": r["test"]["MRE"], "params": r["params"]})
        df_p = pd.DataFrame(rows_p)
        p_csv = os.path.join(PARAM_DIR, "summary.csv")
        df_p.to_csv(p_csv, index=False, float_format="%.6f")
        print(f"[参数汇总] -> {p_csv}", flush=True)

    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()

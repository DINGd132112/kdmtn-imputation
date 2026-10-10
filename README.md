# KDMTN — Knowledge-Guided Dynamic Masking for Missing Time-Series Data Imputation

Anonymous artifact accompanying a submission to **The ACM Web Conference 2027 (WWW '27)**.

This repository contains the reference implementation of **KDMTN**, the training and
evaluation pipeline, and the scripts that regenerate every table and figure in the paper.

---

## 1. What the method does

KDMTN is a knowledge-guided dynamic-masking temporal network for imputing missing
values in multivariate time series. Its central idea is to treat *missingness itself*
as a learnable signal rather than a passive indicator:

| Component | Role |
|---|---|
| **MMKE** — Missing-Mechanism Knowledge Encoder | Jointly encodes observed values, the observation mask, and learnable time/feature embeddings into a position-level *missing-structure knowledge* representation. |
| **KGMG** — Knowledge-Guided Mask Generator | Uses the extracted missing structure to re-mask a subset of *observed* entries in block/point patterns under a masking budget, turning direct reconstruction into context-oriented self-supervised inference. |
| **KAST** — Knowledge-Aware Attention (dual stream) | Diagonally-masked multi-head attention applied along the temporal and variable dimensions separately. |
| **RAMF** — Reliability-Aware Multi-View Fusion | Fuses the multi-view representations with reliability weighting. |
| **Uncertainty-aware dual head** | Predicts both the imputation mean and a calibrated log-variance, optimised with a negative-log-likelihood term. |

Training minimises a joint objective: a masked-imputation loss (MIT) on the
re-masked positions, an observed-reconstruction loss (ORT) on the effective
observed positions, and a Gaussian negative log-likelihood (NLL) for variance
calibration:

```
L = L_MIT + λ_ort · L_ORT + λ_nll · L_NLL
```

The final imputation preserves every originally observed value through a hard
mask-based fusion.

---

## 2. Repository layout

```
.
├── config.yaml                    # dataset list + base hyper-parameters
├── requirements.txt
├── LICENSE
├── data/
│   ├── dataset.py                 # KDMINDataset: PyPOTS-backed loader + missing-rate variants
│   └── __init__.py
├── models/
│   ├── kdmtn_v3.py                # >>> the model that produced all paper results
│   ├── kdmtn.py                   # base KDMTN (MMKE + KGMG + KAST + RAMF composition)
│   ├── mmke.py                    # missing-mechanism knowledge encoder
│   ├── kgmg.py                    # knowledge-guided mask generator
│   ├── kast.py                    # knowledge-aware attention / frequency tokens
│   ├── ramf.py                    # reliability-aware multi-view fusion
│   └── __init__.py
├── train_kdmtn_v3.py              # main training entry point (multi-dataset, multi-rate, multi-seed)
├── ablation_v3.py                 # ablation + hyper-parameter sweep driver
├── compare_baselines_v3.py        # aggregate the baseline comparison into the paper tables
├── make_main_figures.py           # figures 1–5 (bar / line / gain heat-map / param / ablation)
├── make_struct_feat_figure.py     # structural-feature (PCA of Z_m) figure
├── verify_structure_features.py   # quantitative verification of the structural encoding
├── search_hyperparams_v3.py       # hyper-parameter search
└── pypots_baselines.py            # PyPOTS baseline runs (CSDI, PatchTST, SegRNN, …)
```

> **Note.** `models/kdmtn.py` is a *different* architecture from `models/kdmtn_v3.py`.
> The **v3** model is the one evaluated in the paper; the base model is retained only
> because the structural-feature analysis scripts (`make_struct_feat_figure.py`,
> `verify_structure_features.py`) instantiate it directly.

---

## 3. Installation

Python 3.9+ is required.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended. The code selects the device
automatically and falls back to CPU.

---

## 4. Data

All five benchmarks are **public** and are acquired, split, and masked through the
[**PyPOTS**](https://github.com/WenjieDu/PyPOTS) toolbox, which guarantees an identical
data protocol across every compared method:

| Dataset in this code | Paper name | T | N | Notes |
|---|---|---|---|---|
| `etth1` | ETTh1 | 24 | 7 | electricity transformer temperature, hourly |
| `ettm1` | ETTm1 | 24 | 7 | electricity transformer temperature, minutely |
| `italy_air_quality` | Italy-Air-Quality | 24 | 13 | air-quality monitoring network |
| `electricity` | Electricity | 24 | 370 | UCI ElectricityLoadDiagrams |
| `physionet2012` | PhysioNet2012 | 48 | 37 | ICU vital signs (self-supervised setting) |

Datasets are downloaded and cached into `data/` automatically on first use.
Missing-rate variants are stored as `data/<tag>_rate<r>.npz`; delete the cache to
regenerate them. **No data is redistributed with this repository.**

---

## 5. Reproducing the results

### Quick smoke test

```bash
python train_kdmtn_v3.py --smoke
```

### Full main-table run

The paper reports the mean test MAE over three seeds (41/42/43) at five missing
rates (0.1 / 0.3 / 0.5 / 0.7 / 0.9):

```bash
python train_kdmtn_v3.py \
    --datasets etth1,ettm1,electricity,italy_air_quality,physionet2012 \
    --rates 0.1,0.3,0.5,0.7,0.9 \
    --seeds 41,42,43 \
    --d_model 128 --num_layers 2 --num_heads 8 --batch_size 32 \
    --lr 1e-3 --lambda_ort 0.5 --lambda_nll 0.1 --mit_ratio 0.4 --dropout 0.1 \
    --epochs 200 --patience 15
```

These are the defaults, so `python train_kdmtn_v3.py` with the dataset list above
reproduces the paper configuration. `physionet2012` is trained with the
self-supervised loss variant; the remaining datasets use the supervised variant
(refer to `SELFSUP_DATASETS` in `train_kdmtn_v3.py`).

### Baselines

```bash
python pypots_baselines.py        # CSDI, PatchTST, SegRNN, iTransformer, TimeMixer++, TimesNet, Autoformer
python compare_baselines_v3.py    # aggregate into the comparison tables
```

### Ablation study and sensitivity analysis

```bash
python ablation_v3.py --datasets etth1,ettm1 --rate 0.5 --group all
```

### Figures

```bash
python make_main_figures.py          # fig1_bar_5x5, fig2_line_1x5, fig3_improve_5x5, fig4_param, fig5_ablation
python make_struct_feat_figure.py    # fig_struct_feat  (PCA of the MMKE representation Z_m)
python verify_structure_features.py  # numbers quoted in the structural-feature analysis
```

---

## 6. Default configuration

| Hyper-parameter | Value |
|---|---|
| hidden dimension `d` | 128 |
| attention blocks `L` | 2 |
| attention heads `H` | 8 |
| dropout | 0.1 |
| batch size | 32 |
| learning rate | 1e-3 (Adam, ReduceLROnPlateau) |
| weight decay | 1e-5 |
| gradient-norm clipping | 1.0 |
| `λ_ort` | 0.5 |
| `λ_nll` | 0.1 |
| KDM budget `mit_ratio` | 0.4 |
| early stopping | validation MAE, patience 15 |
| seeds | 41 / 42 / 43 |

These settings were found to be a robust operating point across all five datasets
without per-dataset tuning; see the hyper-parameter sensitivity study for details.

---

## 7. Evaluation protocol

Metrics are computed **only on originally missing positions**, except for datasets
that ship an `indicating_mask`, where the protocol follows the PyPOTS convention.
Reported numbers are the mean over the three seeds.

---

## 8. Citation

This artifact is under anonymous review. If you use this code, please cite the
accompanying paper:

```bibtex
@inproceedings{kdmtn2027,
  title     = {KDMTN: Knowledge-Guided Dynamic Masking for Missing Time-Series Data Imputation},
  booktitle = {Proceedings of the ACM Web Conference 2027 (WWW '27)},
  year      = {2027}
}
```

A permanent, citable release archived in a DOI-issuing repository will replace this
anonymous artifact upon publication.

---

## 9. License

Released under the MIT License — see [`LICENSE`](LICENSE).

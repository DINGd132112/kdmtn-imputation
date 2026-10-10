# -*- coding: utf-8 -*-
"""KDMTN-V3: 显式双流缺失插补器 (针对 KDMTN-V2 在 Physio 高缺失场景失利的重构)

设计取舍 (相对 V2, 依据 2026-08-19 架构梳理):
  V2 的问题是 "多视图重建 + 知识门控" —— KGMG 只二次遮挡观测位点、缺失位点无显式
  查询、RAMF 依据观测区误差门控(与缺失区无关)、L_rec 只监督人工遮挡。Physio 训练集
  无真值标签 -> 插补任务与训练任务分布不匹配。

V3 核心 (SAITS/CSDI 关键机制的移植 + KDMTN 缺失结构保留):
  1. 位置级 token: 观测位点 = 值投影([X_obs, M0]) + 时间嵌入 + 特征嵌入;
     缺失位点 = 可学习 per-feature 缺失查询 + 时间嵌入 + 特征嵌入 (显式区分缺失)
  2. 掩码门控注意力: 时间维/变量维自注意力, key-mask = M0 —— 缺失位点只允许
     从观测位点聚合信息 (观测上下文条件化, 防止缺失值噪声传播)
  3. 观测交叉解码: 最后一层 cross-attention 仍以 M0 为 key-mask, 形成
     "缺失查询 -> 观测上下文" 的显式读取路径
  4. 训练目标 (ORT + MIT + NLL):
       L_ORT  : 观测位点保留 (X_final ≈ X_obs on M0)
       L_MIT  : 真实缺失模式二次遮挡恢复 (块状/变量级/点状, 用观测真值监督)
       L_NLL  : 高斯负对数似然 (不确定性校准)
     —— Physio train 无 X_ori, 全自监督; val/test 用 indicating 口径评估
  5. 不确定性输出: mu + log_var, 缺失位点可报告插补置信度

注意: 不使用 MMKE/KGMG/RAMF (经实验证明对缺失插补边际贡献低), M0 本身即缺失结构表达。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMHA(nn.Module):
    """多头自注意力, 支持 key-mask 与 DMSA 对角掩码.

    - key_mask (可选): key_mask=0 的位置不可被 attend. 注意: Physio 观测位点仅 ~20%,
      严格 key-mask 会切断缺失位点间信息传播, 导致输出退化为常数 (实测 val 卡 0.69,
      不如 V2 全互注意力的 0.63) —— 默认关闭.
    - diag_mask (默认开启, SAITS DMSA 式): 禁止位置 attend 自身, 强制从其他位置聚合,
      防止 "抄自己" 的琐碎解.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads, self.d_k = d_model, num_heads, d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_mask=None, diag_mask=False):
        # x: (B, L, d); key_mask: (B, L) bool/float, 1=可作为 key
        B, L, _ = x.shape
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,H,L,L)
        if key_mask is not None:
            km = key_mask[:, None, None, :].float()          # (B,1,1,L)
            scores = scores * km + (1.0 - km) * (-1e9)
        if diag_mask:
            eye = torch.eye(L, device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(eye[None, None], -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


class V3Block(nn.Module):
    """时间维注意力 + 变量维注意力 + FFN.

    use_key_mask=True 时带观测 key-mask (严格观测门控, 对比实验用);
    use_diag=True 时禁止 self-attend (DMSA 式, 默认).
    ablate_t / ablate_v: 组件消融开关 (跳过对应注意力分支, 默认关闭即不消融).
    """
    def __init__(self, d_model, num_heads, dropout=0.1, use_key_mask=False, use_diag=True,
                 ablate_t=False, ablate_v=False):
        super().__init__()
        self.use_key_mask = use_key_mask
        self.use_diag = use_diag
        self.ablate_t = ablate_t
        self.ablate_v = ablate_v
        self.t_attn = MaskedMHA(d_model, num_heads, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.v_attn = MaskedMHA(d_model, num_heads, dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
        self.n3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, M0):
        # x: (B,T,N,d); M0: (B,T,N)
        B, T, N, _ = x.shape
        # -- 时间维: 每个变量是一条长度为 T 的序列 (B*N, T, d)
        xt = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        if not self.ablate_t:
            km_t = M0.permute(0, 2, 1).reshape(B * N, T) if self.use_key_mask else None
            xt = xt + self.drop(self.t_attn(self.n1(xt), key_mask=km_t, diag_mask=self.use_diag))
        # -- 变量维: 每个时间步是一条长度为 N 的序列 (B*T, N, d)
        xv = xt.reshape(B, N, T, -1).permute(0, 2, 1, 3).reshape(B * T, N, -1)
        if not self.ablate_v:
            km_v = M0.reshape(B * T, N) if self.use_key_mask else None
            xv = xv + self.drop(self.v_attn(self.n2(xv), key_mask=km_v, diag_mask=self.use_diag))
        x = xv.reshape(B, T, N, -1)
        x = self.n3(x + self.drop(self.ffn(x)))
        return x


class ObservedDecoder(nn.Module):
    """观测交叉解码: key-mask 关闭时退化为普通注意力残差层 (默认); 开启时缺失查询显式读取观测上下文."""
    def __init__(self, d_model, num_heads, dropout=0.1, use_key_mask=False, use_diag=True,
                 ablate_t=False, ablate_v=False):
        super().__init__()
        self.use_key_mask = use_key_mask
        self.use_diag = use_diag
        self.ablate_t = ablate_t
        self.ablate_v = ablate_v
        self.t_attn = MaskedMHA(d_model, num_heads, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.v_attn = MaskedMHA(d_model, num_heads, dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, M0):
        B, T, N, _ = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        if not self.ablate_t:
            km_t = M0.permute(0, 2, 1).reshape(B * N, T) if self.use_key_mask else None
            xt = xt + self.drop(self.t_attn(self.n1(xt), key_mask=km_t, diag_mask=self.use_diag))
        xv = xt.reshape(B, N, T, -1).permute(0, 2, 1, 3).reshape(B * T, N, -1)
        if not self.ablate_v:
            km_v = M0.reshape(B * T, N) if self.use_key_mask else None
            xv = xv + self.drop(self.v_attn(self.n2(xv), key_mask=km_v, diag_mask=self.use_diag))
        return xv.reshape(B, T, N, -1)


class KDMTN_V3(nn.Module):
    """KDMTN-V3: 显式双流缺失插补器.

    输入: X_obs (B,T,N) 观测值(缺失处预填补, 不影响建模), M0 (B,T,N) 观测掩码
    输出: mu (B,T,N) 插补均值, log_var (B,T,N) 对数方差, X_final = M0*X_obs + (1-M0)*mu
    """
    def __init__(self, T, N, d_model=128, num_layers=2, num_heads=4, dropout=0.1,
                 use_key_mask=False, use_diag=True, use_obs_gate=False,
                 use_mask_emb=False, ablate_t=False, ablate_v=False,
                 ablate_pos=False, ablate_dec=False, ablate_unc=False):
        super().__init__()
        self.T, self.N, self.d_model = T, N, d_model
        # 消融开关 (默认全部关闭 -> 完整模型)
        self.ablate_pos = ablate_pos
        self.ablate_dec = ablate_dec
        self.ablate_unc = ablate_unc
        # 值投影: 输入 [观测值(缺失处0填充), 观测掩码] —— SAITS 式双通道.
        # 关键: 缺失位点的值通道必须为 0, 模型无法从"缺失标记"解码出"值",
        # 只能从观测上下文推断 —— 避免 mask 嵌入提供的"输出常数"捷径
        # (V3 初版用 mask_emb 替换 token, 实测 head 可直接解码常数, val 卡 0.70)
        self.val_proj = nn.Sequential(nn.Linear(2, d_model), nn.GELU(),
                                      nn.Linear(d_model, d_model))
        # 位置/身份嵌入
        self.time_emb = nn.Parameter(torch.randn(T, d_model) * 0.02)
        self.feat_emb = nn.Parameter(torch.randn(N, d_model) * 0.02)
        # 可学习缺失查询 (per-feature) —— 可选, 默认关闭 (对比实验)
        self.use_mask_emb = use_mask_emb
        if use_mask_emb:
            self.mask_emb = nn.Parameter(torch.randn(N, d_model) * 0.02)
        # 双流编码 + 观测解码
        self.layers = nn.ModuleList([V3Block(d_model, num_heads, dropout,
                                             use_key_mask=use_key_mask, use_diag=use_diag,
                                             ablate_t=ablate_t, ablate_v=ablate_v)
                                     for _ in range(num_layers)])
        self.decoder = ObservedDecoder(d_model, num_heads, dropout,
                                       use_key_mask=use_key_mask, use_diag=use_diag,
                                       ablate_t=ablate_t, ablate_v=ablate_v)
        # 输出头: 均值 + 对数方差
        self.head_mu = nn.Linear(d_model, 1)
        self.head_logvar = nn.Linear(d_model, 1)
        # 残差门控 (观测位点强制保留真值, 缺失位点走生成)
        self.use_obs_gate = use_obs_gate
        self.out_gate = nn.Linear(d_model, 1)

        self.reset_params()

    def reset_params(self):
        nn.init.normal_(self.head_mu.weight, std=0.02)
        nn.init.zeros_(self.head_mu.bias)
        nn.init.zeros_(self.head_logvar.weight)
        nn.init.constant_(self.head_logvar.bias, -2.0)   # 初始方差较小, 避免 NLL 早期主导

    def forward(self, X_obs, M0, return_all=False):
        B, T, N = X_obs.shape
        # 1) 位置 token: 双通道输入 [值(缺失处0), 掩码] + 位置嵌入.
        #    缺失位点不替换值 —— 值通道保持 0, 模型必须从观测上下文推断缺失值
        X_in = torch.stack([X_obs * M0, M0], dim=-1)   # (B,T,N,2), 缺失位点值通道为 0
        tok = self.val_proj(X_in)
        if self.use_mask_emb:
            # 可选: 额外叠加可学习缺失查询 (SAITS mask embedding 只作标记, 不替换值)
            tok = tok + self.mask_emb[None, None, :, :] * (1.0 - M0).unsqueeze(-1)
        if not self.ablate_pos:
            tok = tok + self.time_emb[None, :, None, :] + self.feat_emb[None, None, :, :]

        x = tok
        for layer in self.layers:
            x = layer(x, M0)
        if not self.ablate_dec:
            x = self.decoder(x, M0)

        mu = self.head_mu(x).squeeze(-1)              # (B,T,N)
        if self.ablate_unc:
            # 消融不确定性分支: 固定方差 (不经过 head_logvar, 该参数不参与训练)
            log_var = torch.full_like(mu, -2.0)
        else:
            log_var = self.head_logvar(x).squeeze(-1)     # (B,T,N), 使用时 clamp
        if self.use_obs_gate:
            g = torch.sigmoid(self.out_gate(x)).squeeze(-1)
            # 观测位点直接保留真值, 缺失位点以生成值为主 (门控微调边界)
            X_final = g * X_obs + (1.0 - g) * mu
            X_final = M0 * X_obs + (1.0 - M0) * X_final
            gate_out = g
        else:
            # 无门控: 观测保留 + 缺失生成 (最简单可靠的合并)
            X_final = M0 * X_obs + (1.0 - M0) * mu
            gate_out = None

        if return_all:
            return X_final, {'mu': mu, 'log_var': log_var, 'gate': gate_out, 'tok': tok}
        return X_final


def build_mit_mask(M0, mit_ratio=0.4, n_blocks=2, block_len=(2, 12), p_point=0.1, seed=None):
    """真实缺失模式二次遮挡掩码 (仅观测位点, 用于 MIT 自监督).

    模式构成: 连续时间段块状缺失(多变量子集) + 点状随机缺失 —— 模拟 Physio 的
    真实缺失结构 (长连续段、变量同步缺失), 而非单纯 i.i.d. 点缺失.
    总量控制: 遮挡量 ~ mit_ratio * 观测位点数 (Physio 观测仅 ~20%, 防止输入全空).
    """
    B, T, N = M0.shape
    dev = M0.device
    rng = torch.Generator(device=dev)
    if seed is not None:
        rng.manual_seed(seed)
    M = torch.zeros(B, T, N, device=dev)
    # 块状: 每样本 n_blocks 个连续块 (随机起点/长度/变量子集)
    for b in range(B):
        for _ in range(n_blocks):
            t0 = torch.randint(0, T, (1,), generator=rng, device=dev).item()
            ln = int(torch.randint(block_len[0], block_len[1] + 1, (1,), generator=rng, device=dev).item())
            t1 = min(t0 + ln, T)
            nv = int(torch.randint(1, max(2, N // 4 + 1), (1,), generator=rng, device=dev).item())
            vs = torch.randperm(N, generator=rng, device=dev)[:nv]
            M[b, t0:t1, vs] = 1.0
    # 点状随机
    pr = torch.rand(B, T, N, generator=rng, device=dev)
    M = torch.clamp(M + (pr < p_point).float(), max=1.0) * M0
    # 总量裁剪到 mit_ratio * 观测位点数
    cap = int(mit_ratio * M0.sum().item())
    cur = int(M.sum().item())
    if cur > cap:
        flat = M.flatten()
        perm = torch.randperm(flat.numel(), generator=rng, device=dev)
        flat[perm[:cur - cap]] = 0.0
        M = flat.reshape(B, T, N)
    return M


def v3_loss(mu, log_var, X_final, M_eff, X_obs, M_mit, M0, lambda_ort=0.5, lambda_nll=0.1):
    """SAITS 式 ORT + MIT + NLL (Physio train 无真值, 全用观测值自监督).

    M_eff = M0*(1-M_mit): 保留观测位点 (训练输入中可见)
    M_mit : 二次遮挡位点 (训练输入中已被抹掉观测值, 模型必须从上下文推断)
    关键: M_mit 位点在模型输入中不可见 (=0 填充), 因此 L_mit 强制模型真正从观测
    上下文推断缺失值, 而不是复制输入 —— 修复 V3 之前 bug: 未遮蔽输入导致模型
    输出常数即可最小化损失, 卡 trivial 0.698.
    """
    eps = 1e-8
    n_ort = M_eff.sum() + eps
    n_mit = M_mit.sum() + eps
    # ORT: 保留观测位点, 内部表示 mu 应接近观测值 (观测不被破坏)
    L_ort = (torch.abs(mu - X_obs) * M_eff).sum() / n_ort
    # MIT: 二次遮挡位点恢复 (模型看不到这些位点, 必须推断)
    L_mit = (torch.abs(mu - X_obs) * M_mit).sum() / n_mit
    # NLL: 高斯负对数似然 (不确定性校准), 监督位点同 MIT
    lv = log_var.clamp(-6.0, 6.0)
    var = torch.exp(lv)
    nll = 0.5 * (((X_obs - mu) ** 2) / (var + eps) + lv + math.log(2 * math.pi))
    L_nll = (nll * M_mit).sum() / n_mit
    total = L_mit + lambda_ort * L_ort + lambda_nll * L_nll
    return total, {'L_ort': L_ort.item(), 'L_mit': L_mit.item(),
                   'L_nll': L_nll.item(), 'total': total.item()}


def v3_loss_supervised(mu, log_var, X_final, X_true, M_eff, lambda_ort=0.5, lambda_nll=0.1):
    """真值监督版 (用于有完整真值的数据集, 如 etth1/ettm1/electricity/italy/beijing).

    这些数据集 X_ori 完整 (缺失 0%), 因此可直接在所有"被抹掉"的位点 (原始缺失 +
    MIT 二次遮挡) 用真值 X_true 监督, 比 Physio 的纯自监督 ORT+MIT 更强.

    M_eff = M0*(1-M_mit): 训练输入中可见的保留观测位点
    M_miss = 1 - M_eff: 原始缺失位点 + MIT 二次遮挡位点 (输入中不可见, 用真值监督)
    """
    eps = 1e-8
    M_miss = 1.0 - M_eff
    n_miss = M_miss.sum() + eps
    n_ort = M_eff.sum() + eps
    # 缺失/MIT 位点: 用真值监督 (核心插补目标)
    L_final = (torch.abs(mu - X_true) * M_miss).sum() / n_miss
    # ORT: 保留观测位点, 内部表示 mu 接近真值 (观测不被破坏)
    L_ort = (torch.abs(mu - X_true) * M_eff).sum() / n_ort
    # NLL: 高斯负对数似然 (不确定性), 全部有真值位点
    lv = log_var.clamp(-6.0, 6.0)
    var = torch.exp(lv)
    nll = 0.5 * (((X_true - mu) ** 2) / (var + eps) + lv + math.log(2 * math.pi))
    n_all = (M_miss + M_eff).sum() + eps
    L_nll = (nll * (M_miss + M_eff)).sum() / n_all
    total = L_final + lambda_ort * L_ort + lambda_nll * L_nll
    return total, {'L_final': L_final.item(), 'L_ort': L_ort.item(),
                   'L_nll': L_nll.item(), 'total': total.item()}
"""Knowledge-Guided Dynamic Mask Generator (KGMG)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


_TRIU_CACHE = {}


def _get_triu_indices(N, device):
    key = (N, str(device))
    idx = _TRIU_CACHE.get(key)
    if idx is None:
        idx = torch.triu_indices(N, N, offset=1, device=device)
        _TRIU_CACHE[key] = idx
    return idx


def compute_structure_statistics(M):
    """缺失结构统计: 缺失率 / 缺失长度直方图 / 变量间缺失相关 (全向量化实现).

    与原(双循环)版本结果完全一致, 但避免 BxN 次 GPU kernel 启动, 提速 2 个数量级.
    """
    B, T, N = M.shape
    device = M.device
    missing_rate = (1.0 - M).mean(dim=(1, 2))

    # ---- 缺失长度直方图 (向量化 RLE) ----
    bins = torch.linspace(0, T, 11, device=device)
    D_missing = 1.0 - M                                   # (B, T, N)
    padded = F.pad(D_missing, (0, 0, 1, 1), mode="constant", value=0.0)  # (B, T+2, N)
    diff = padded[:, 1:, :] - padded[:, :-1, :]           # (B, T+1, N)
    starts = (diff == 1.0)
    ends = (diff == -1.0)
    B2 = B * N
    starts_f = starts.transpose(1, 2).reshape(B2, T + 1)  # (B*N, T+1)
    ends_f = ends.transpose(1, 2).reshape(B2, T + 1)
    seg_id = torch.cumsum(starts_f.long(), dim=1)         # 每个位置所属缺失段 id
    start_idx = starts_f.nonzero()
    if start_idx.numel() == 0:
        missing_length_hist = torch.zeros(B, 10, device=device)
    else:
        end_idx = ends_f.nonzero()
        max_seg = int(seg_id.max().item())
        start_tab = torch.full((B2, max_seg + 1), -1, dtype=torch.long, device=device)
        end_tab = torch.full_like(start_tab, -1)
        start_tab[start_idx[:, 0], seg_id[start_idx[:, 0], start_idx[:, 1]]] = start_idx[:, 1]
        end_tab[end_idx[:, 0], seg_id[end_idx[:, 0], end_idx[:, 1]]] = end_idx[:, 1]
        valid = (start_tab >= 0) & (end_tab >= 0)
        lengths = (end_tab - start_tab).clamp(min=0)
        hist_idx = torch.bucketize(lengths, bins) - 1
        hist_idx = hist_idx.clamp(0, 9)
        hist_flat = torch.zeros(B2, 10, device=device)
        hist_flat.scatter_add_(1, hist_idx, valid.float())
        missing_length_hist = hist_flat.reshape(B, N, 10).sum(dim=1) / max(N, 1)

    # ---- 缺失模式变量间相关 (已向量化, 缓存 triu 索引) ----
    D = 1.0 - M
    D_centered = D - D.mean(dim=1, keepdim=True)
    D_std = D.std(dim=1, keepdim=True) + 1e-8
    D_norm = D_centered / D_std
    corr_mat = torch.bmm(D_norm.transpose(1, 2), D_norm) / (T - 1)
    triu_indices = _get_triu_indices(N, device)
    var_corr_triu = corr_mat[:, triu_indices[0], triu_indices[1]]
    return missing_rate, missing_length_hist, var_corr_triu


class KnowledgeGuidedMaskGenerator(nn.Module):
    def __init__(self, T, N, d_model):
        super().__init__()
        self.T, self.N, self.d_model = T, N, d_model
        self.mask_mlp = nn.Sequential(
            nn.Linear(1 + d_model, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, X_obs, Zm, M0, K):
        B, T, N = X_obs.shape
        X_exp = X_obs.unsqueeze(-1)
        Zm_exp = Zm.unsqueeze(2).expand(-1, -1, N, -1)
        combined = torch.cat([X_exp, Zm_exp], dim=-1)
        logits = self.mask_mlp(combined).squeeze(-1)
        S0 = compute_structure_statistics(M0)
        masks, M_tildes, P_masks = [], [], []
        for k in range(K):
            if self.training:
                noise = torch.rand_like(logits).clamp(min=1e-10, max=1.0 - 1e-10)
                gumbel_noise = -torch.log(-torch.log(noise))
                y_logits = logits + gumbel_noise
                P_mask = torch.sigmoid(y_logits)
                M_tilde_hard = (P_mask > 0.5).float()
                M_tilde = M_tilde_hard.detach() + P_mask - P_mask.detach()
            else:
                P_mask = torch.sigmoid(logits)
                M_tilde = (P_mask > 0.5).float()
            M_tilde = M_tilde * M0
            Mk = M0 * (1.0 - M_tilde)
            masks.append(Mk); M_tildes.append(M_tilde); P_masks.append(P_mask)
        masks = torch.stack(masks, dim=1)
        M_tildes = torch.stack(M_tildes, dim=1)
        P_masks = torch.stack(P_masks, dim=1)
        return masks, M_tildes, P_masks, S0

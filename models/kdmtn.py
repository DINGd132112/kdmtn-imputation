"""K-DMTN: Knowledge-guided Dynamic Masking Transformer Network.

支持消融开关 (默认全部启用, 数值与原实现一致):
    use_mmke:  缺失机制知识编码器
    use_kgmg:  知识引导动态掩码生成器 (False 时用固定随机掩码)
    use_freq:  KAST 频率 token 分支
    use_bias:  KAST 知识引导注意力偏置
    use_ramf:  可靠性多视图融合 (False 时多视图均匀平均)
"""
import torch
import torch.nn as nn

from models.mmke import MissingMechanismKnowledgeEncoder
from models.kgmg import KnowledgeGuidedMaskGenerator, compute_structure_statistics
from models.kast import KAST
from models.ramf import ReliabilityAwareMultiViewFusion


class KDMTN(nn.Module):
    def __init__(self, T, N, d_model=64, num_layers=2, num_heads=4, K=3, top_Kf=8,
                 use_mmke=True, use_kgmg=True, use_freq=True, use_bias=True, use_ramf=True):
        super().__init__()
        self.T, self.N, self.d_model, self.K = T, N, d_model, K
        self.use_mmke = use_mmke
        self.use_kgmg = use_kgmg
        self.use_ramf = use_ramf
        self.mmke = MissingMechanismKnowledgeEncoder(N, T, d_model)
        self.kgmg = KnowledgeGuidedMaskGenerator(T, N, d_model)
        self.kast = KAST(T, N, d_model, num_layers, num_heads, top_Kf,
                         use_freq=use_freq, use_bias=use_bias)
        self.ramf = ReliabilityAwareMultiViewFusion(d_model, K)
        if not use_mmke:
            # 无知识编码: 用简单线性投影替代 (输入 X_obs 与缺失掩码的拼接)
            self.plain_proj = nn.Linear(N * 2, d_model)

    def _get_masks(self, X_obs, Zm, M0):
        """掩码生成: 知识引导 (KGMG) 或固定随机掩码 (w/o KGMG 消融)"""
        if self.use_kgmg:
            return self.kgmg(X_obs, Zm, M0, K=self.K)
        B, T, N = X_obs.shape
        masks, M_tildes, P_masks = [], [], []
        for _ in range(self.K):
            if self.training:
                rand = torch.rand_like(M0)
                M_tilde = (rand > 0.5).float() * M0
            else:
                M_tilde = (M0 * 0.5 > 0.25).float() * M0  # 确定性: 遮 ~50% 观测点
            Mk = M0 * (1.0 - M_tilde)
            masks.append(Mk); M_tildes.append(M_tilde); P_masks.append(torch.full_like(M0, 0.5))
        S0 = compute_structure_statistics(M0)
        return (torch.stack(masks, 1), torch.stack(M_tildes, 1),
                torch.stack(P_masks, 1), S0)

    def forward(self, X_obs, M0, return_all=False):
        B = X_obs.shape[0]
        if self.use_mmke:
            Zm = self.mmke(X_obs, M0)
        else:
            Zm = self.plain_proj(torch.cat([X_obs, M0], dim=-1))

        masks, M_tildes, P_masks, S0 = self._get_masks(X_obs, Zm, M0)

        reconstructions = []
        for k in range(self.K):
            masked_X = X_obs * masks[:, k]
            recon = self.kast(masked_X, Zm)
            reconstructions.append(recon)

        errors, obs_ratios = [], []
        for k in range(self.K):
            observed = masks[:, k]
            obs_count = observed.sum(dim=(1, 2))
            obs_ratio = obs_count / (self.T * self.N)
            recon_error = torch.abs(reconstructions[k] - X_obs) * observed
            error_per_sample = recon_error.sum(dim=(1, 2)) / (obs_count + 1e-8)
            errors.append(error_per_sample)
            obs_ratios.append(obs_ratio)

        errors = torch.stack(errors, dim=1)
        obs_ratios = torch.stack(obs_ratios, dim=1)
        Zm_pooled = Zm.mean(dim=1)
        if self.use_ramf:
            X_final, weights = self.ramf(reconstructions, errors, obs_ratios, Zm_pooled)
        else:
            # w/o RAMF: 多视图均匀平均 (去掉可靠性门控)
            reconstructions_stack = torch.stack(reconstructions, dim=1)
            weights = torch.ones(B, self.K, device=X_obs.device) / self.K
            X_final = reconstructions_stack.mean(dim=1)

        if return_all:
            return X_final, {'Zm': Zm, 'masks': masks, 'M_tildes': M_tildes,
                              'P_masks': P_masks, 'S0': S0,
                              'reconstructions': reconstructions, 'weights': weights,
                              'errors': errors}
        return X_final

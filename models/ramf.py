"""Reliability-Aware Multi-View Fusion (RAMF)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityAwareMultiViewFusion(nn.Module):
    def __init__(self, d_model, K):
        super().__init__()
        self.K, self.d_model = K, d_model
        gate_input_dim = K * 2 + d_model
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, 32), nn.ReLU(),
            nn.Linear(32, K),
        )

    def forward(self, reconstructions, errors, obs_ratios, Zm_pooled):
        gate_input = torch.cat([errors, obs_ratios, Zm_pooled], dim=-1)
        logits = self.gate_net(gate_input)
        weights = F.softmax(logits, dim=-1)
        reconstructions_stack = torch.stack(reconstructions, dim=1)
        X_final = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * reconstructions_stack, dim=1)
        return X_final, weights

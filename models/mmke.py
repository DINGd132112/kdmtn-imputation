"""Missing Mechanism Knowledge Encoder (MMKE)."""
import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    def __init__(self, input_channels, d_t, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, d_t, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(d_t, d_t, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class VariableCNN(nn.Module):
    def __init__(self, input_channels, d_v, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, d_v, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(d_v, d_v, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = x.mean(dim=1, keepdim=True)
        return x


class ContextMLP(nn.Module):
    def __init__(self, input_dim, d_c):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_c * 2), nn.ReLU(),
            nn.Linear(d_c * 2, d_c), nn.ReLU(),
        )
    def forward(self, x):
        return self.mlp(x)


class MissingMechanismKnowledgeEncoder(nn.Module):
    def __init__(self, N, T, d_model=64, d_t=64, d_v=64, d_c=64):
        super().__init__()
        self.N, self.T, self.d_model = N, T, d_model
        self.temporal_cnn = TemporalCNN(input_channels=2*N, d_t=d_t)
        self.variable_cnn = VariableCNN(input_channels=T, d_v=d_v)
        self.context_mlp = ContextMLP(input_dim=T*N*2, d_c=d_c)
        self.proj_t = nn.Linear(d_t, d_model)
        self.proj_v = nn.Linear(d_v, d_model)
        self.proj_c = nn.Linear(d_c, d_model)

    def forward(self, X_obs, M0):
        B, T, N = X_obs.shape
        tem_in = torch.cat([X_obs, M0], dim=-1)
        Kt = self.proj_t(self.temporal_cnn(tem_in))
        # 缺失感知修复: 变量维 CNN 原只用 X_obs(缺失处=0 无缺失感知), 现注入 M0 通道
        Kv = self.proj_v(self.variable_cnn(torch.cat([X_obs, M0], dim=-1))).expand(-1, T, -1)
        ctx_in = torch.cat([X_obs, M0], dim=-1).reshape(B, -1)
        Kc = self.proj_c(self.context_mlp(ctx_in)).unsqueeze(1).expand(-1, T, -1)
        return Kt + Kv + Kc

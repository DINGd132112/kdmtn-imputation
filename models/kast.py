"""Knowledge-Aware SpatioTemporal Transformer (KAST)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrequencyTokenExtractor(nn.Module):
    def __init__(self, T, N, d_model, Kf=8):
        super().__init__()
        self.T, self.N, self.Kf, self.d_model = T, N, Kf, d_model
        self.proj = nn.Linear(N, d_model)

    def forward(self, X_obs):
        B, T, N = X_obs.shape
        X_fft = torch.fft.fft(X_obs, dim=1)
        amplitude = torch.abs(X_fft)[:, :T//2+1, :]
        importance = amplitude.mean(dim=-1)
        topk_indices = torch.topk(importance, min(self.Kf, T//2+1), dim=1).indices
        topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, N)
        top_amplitudes = torch.gather(amplitude, 1, topk_indices)
        return self.proj(top_amplitudes)


class KnowledgeAwareAttentionBias(nn.Module):
    def __init__(self, d_model, T, N):
        super().__init__()
        self.T, self.N, self.d_model = T, N, d_model
        self.proj_t = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.W_t = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_v = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

    def forward(self, Zm):
        B = Zm.shape[0]
        Zm_t = self.proj_t(Zm)
        B_t = torch.bmm(Zm_t, torch.bmm(self.W_t.unsqueeze(0).expand(B, -1, -1), Zm_t.transpose(1, 2)))
        Zm_pooled = Zm.mean(dim=1)
        Zm_v = self.proj_v(Zm_pooled)
        bias_scalar = torch.bmm(Zm_v.unsqueeze(1), torch.bmm(self.W_v.unsqueeze(0).expand(B, -1, -1), Zm_v.unsqueeze(2)))
        B_v = bias_scalar.expand(-1, self.N, self.N) / math.sqrt(self.d_model)
        return B_t, B_v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, bias=None):
        B, L, _ = x.shape
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if bias is not None:
            scores = scores + bias.unsqueeze(1)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


class KASTEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, T, N, d_ff=None):
        super().__init__()
        self.T, self.N = T, N
        d_ff = d_ff or d_model * 4
        self.temporal_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, B_t, B_v):
        B, L, D = x.shape
        T = self.T
        x_time = x[:, :T, :]
        x_freq = x[:, T:, :]
        attn_time = self.temporal_attn(x_time, bias=B_t)
        x_time = self.norm1(x_time + attn_time)
        x = torch.cat([x_time, x_freq], dim=1)
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class KAST(nn.Module):
    def __init__(self, T, N, d_model=64, num_layers=2, num_heads=4, top_Kf=8,
                 use_freq=True, use_bias=True):
        super().__init__()
        self.T, self.N, self.d_model, self.num_layers = T, N, d_model, num_layers
        self.use_freq = use_freq
        self.use_bias = use_bias
        self.freq_extractor = FrequencyTokenExtractor(T, N, d_model, Kf=top_Kf)
        self.input_proj = nn.Linear(N, d_model)
        self.bias_gen = KnowledgeAwareAttentionBias(d_model, T, N)
        self.pos_encoder = PositionalEncoding(d_model, T)
        self.layers = nn.ModuleList([KASTEncoderLayer(d_model, num_heads, T, N) for _ in range(num_layers)])
        self.output_proj = nn.Linear(d_model, N)

    def forward(self, masked_X, Zm):
        B, T, N = masked_X.shape
        H = self.pos_encoder(self.input_proj(masked_X))
        if self.use_freq:
            Zf = self.freq_extractor(masked_X)
            H0 = torch.cat([H, Zf], dim=1)
        else:
            H0 = H
        if self.use_bias:
            B_t, B_v = self.bias_gen(Zm)
        else:
            B_t = torch.zeros(B, T, T, device=masked_X.device)
            B_v = torch.zeros(B, N, N, device=masked_X.device)
        x = H0
        for layer in self.layers:
            x = layer(x, B_t, B_v)
        reconstructed = self.output_proj(x[:, :T, :])
        return reconstructed

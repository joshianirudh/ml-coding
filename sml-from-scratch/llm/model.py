import torch
from torch import nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, causal_mask=False, n_heads=4, dim = 768):
        super().__init__()
        self.causal_mask = causal_mask
        self.n_heads = n_heads
        self.dim = dim
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, q, k, v):
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        batch_size, seq_len, _ = q.shape
        head_dims = self.dim // self.n_heads
        scale = head_dims ** -0.5
        # b * s * d -> b * h * s * h_d
        # steps:
        # 1) break d: b * s * h * h_d
        # 2) transpose: b * h * s * h_d
        # do self_attn on each head
        q = q.view(batch_size, seq_len, self.n_heads, head_dims).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, head_dims).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, head_dims).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if self.causal_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = torch.matmul(attn, v)
        # b * h * s * h_d -> b * s * d
        attn = attn.transpose(1, 2).view(batch_size, seq_len, self.dim)
        return self.out_proj(attn)

class FFN(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        hidden_dim = 4 * dim
        self.layer1 = nn.Linear(dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, dim)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

"""
Assuming postLN architecture from the classic transformer
"""
class Encoder(nn.Module):
    def __init__(self, n_heads=4, dim=768):
        super().__init__()
        self.dim = dim
        self.mha = MHA(causal_mask=False, n_heads = n_heads, dim=dim)
        self.norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        attn = self.mha.forward(x, x, x)
        x = x + attn
        ffn = self.ffn.forward(self.norm(x))
        x = x + ffn
        return self.norm2(x)



class Decoder(nn.Module):
    def __init__(self, n_heads=4, dim=768):
        super().__init__()
        self.mha = MHA(causal_mask=True, n_heads = n_heads, dim=dim)
        self.norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn = self.mha.forward(x, x, x)
        x = x + attn
        x = self.norm(x)
        ffn = self.ffn.forward(x)
        x = x + ffn
        return self.norm2(x)

class CrossAttentionDecoder(nn.Module):
    def __init__(self, n_heads=4, dim=768):
        super().__init__()
        self.m_mha = MHA(causal_mask=True, n_heads = n_heads, dim=dim)
        self.norm = nn.LayerNorm(dim)
        self.mha = MHA(causal_mask=False, n_heads = n_heads, dim=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim)
        self.norm3 = nn.LayerNorm(dim)
    def forward(self, x, k, v):
        masked_mha =  self.m_mha.forward(x, x, x)
        x = x + masked_mha
        x = self.norm(x)
        mha = self.mha.forward(x, k, v)
        x = x + mha
        x = self.norm2(x)
        ffn = self.ffn.forward(x)
        x = x + ffn
        return self.norm3(x)

class LLM(nn.Module):
    def __init__(self, n_encoders, n_decoders, n_heads, dim):
        super().__init__()
        # encoder-decoder architecture
        if n_encoders >0:
            self.encoders = nn.ModuleList([Encoder(n_heads, dim) for i in range(n_encoders)])
            if n_decoders >0:
                self.decoders = nn.ModuleList([CrossAttentionDecoder(n_heads, dim) for i in range(n_decoders)])
        # decoder only architecture
        elif n_decoders>0:
            self.decoders = nn.ModuleList([Decoder(n_heads, dim) for i in range(n_decoders)])
    def forward(self):
        pass

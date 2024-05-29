import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import math
from utils import *


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SELayer(nn.Module):
    def __init__(self, c, r=4, use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, s, h = x.shape
        y = self.squeeze(x).view(bs, s)
        y = self.excitation(y).view(bs, s, 1)
        return x * y.expand_as(x)


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        y = self.linear2(self.dropout(self.activation(self.linear1(self.norm(x)))))
        y = x + y
        return y


class TemporalSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head

        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)

        y = x + y
        return y

    
class TemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 skip=False,
                 se_dim=21,
                 se_r=4
                 ):
        super().__init__()
        if skip:
            self.skip_linear = nn.Linear(2 * latent_dim, latent_dim)

        self.sa_block = TemporalSelfAttention(
            latent_dim, num_head, dropout)
        self.ffn = FFN(latent_dim, ffn_dim, dropout)
        self.se = SELayer(se_dim, r=se_r, use_max_pooling=False)

    def forward(self, x, skip=None):
        if skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.se(x)
        x = self.sa_block(x)
        x = self.ffn(x)
        return x


class MotionTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames + 2, latent_dim))

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        
        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.in_blocks = nn.ModuleList([
            TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,)
            for i in range(self.num_layers // 2)])

        self.mid_block = TemporalDiffusionTransformerDecoderLayer(
                                latent_dim=latent_dim,
                                time_embed_dim=self.time_embed_dim,
                                ffn_dim=ff_size,
                                num_head=num_heads,
                                dropout=dropout,)

        self.out_blocks = nn.ModuleList([
            TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                    skip=True)
            for i in range(self.num_layers // 2)])
                
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, mod=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)).unsqueeze(1)

        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1)).unsqueeze(1)
            emb = emb + mod_proj

        h = self.joint_embed(x)
        h = torch.cat([emb, h], dim=1)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T+1, :]

        skips = []
            
        for blk in self.in_blocks:
            h = blk(h)
            skips.append(h)
        
        h = self.mid_block(h)

        for blk in self.out_blocks:
            h = blk(h, skips.pop())

        output = self.out(h[:, 1:, :]).view(B, T, -1).contiguous()
        return output

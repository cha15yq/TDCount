from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from timm.models.layers import DropPath


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.1,
                 proj_drop_ratio=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x_q, x_kv):
        assert x_q.shape == x_kv.shape
        B, N, C = x_q.shape

        kv = self.kv(x_kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = self.q(x_q).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.1,
                 attn_drop_ratio=0.1,
                 drop_path_ratio=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(CrossAttentionBlock, self).__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x[0] + self.drop_path1(self.attn(self.norm1_q(x[0]), self.norm1_kv(x[1])))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class CrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim=256, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.1, attn_drop_ratio=0.1, drop_path_ratio=0.1,
                 norm_layer=None,act_layer=None):
        super(CrossAttentionTransformer, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.norm = norm_layer(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            CrossAttentionBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                                norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

    def forward(self, modal_x, modal_y):
        assert modal_x.shape == modal_y.shape
        B, C, H, W = modal_x.shape

        feature_modal_x = modal_x.flatten(2).transpose(1, 2)
        feature_modal_y = modal_y.flatten(2).transpose(1, 2)

        x = self.blocks((feature_modal_x, feature_modal_y))
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

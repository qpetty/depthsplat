import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
XFORMERS_AVAILABLE = False

try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind
        XFORMERS_AVAILABLE = True
except ImportError:
    pass


def use_pytorch_attention():
    """Check if we should use PyTorch native attention (for TensorRT compatibility)."""
    return os.environ.get("FORCE_PYTORCH_ATTENTION") == "1"


class CrossAttention(nn.Module):
    def __init__(
        self, 
        in_dim1,
        in_dim2,
        dim=128,
        out_dim=None,
        num_heads=4,
        qkv_bias=False,
        proj_bias=False,
    ):
        super().__init__()

        # Only require xformers if not forcing PyTorch attention
        if not use_pytorch_attention():
            assert XFORMERS_AVAILABLE, "xFormers required. Set FORCE_PYTORCH_ATTENTION=1 for TensorRT."

        if out_dim is None:
            out_dim = in_dim1

        self.num_heads = num_heads
        self.dim = dim
        self.q = nn.Linear(in_dim1, dim, bias=qkv_bias)
        self.kv = nn.Linear(in_dim2, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, out_dim, bias=proj_bias)

    def forward(self, x, y):
        c = self.dim
        b, n1, c1 = x.shape
        n2, c2 = y.shape[1:]

        q = self.q(x).reshape(b, n1, self.num_heads, c // self.num_heads)
        kv = self.kv(y).reshape(b, n2, 2, self.num_heads, c // self.num_heads)
        
        if use_pytorch_attention() or not XFORMERS_AVAILABLE:
            # PyTorch native attention for TensorRT compatibility
            k, v = kv.unbind(2)
            
            # xformers: [b, seq, heads, dim] -> PyTorch: [b, heads, seq, dim]
            q_pt = q.permute(0, 2, 1, 3)
            k_pt = k.permute(0, 2, 1, 3)
            v_pt = v.permute(0, 2, 1, 3)
            
            x = F.scaled_dot_product_attention(q_pt, k_pt, v_pt)
            x = x.permute(0, 2, 1, 3)  # back to [b, seq, heads, dim]
        else:
            # xformers for normal inference (faster)
            k, v = unbind(kv, 2)
            x = memory_efficient_attention(q, k, v)
        
        x = x.reshape(b, n1, c)
        x = self.proj(x)
        return x
        

class UNetCrossAttentionBlock(nn.Module):
    def __init__(self,
        in_dim1,
        in_dim2,
        dim=128,
        out_dim=None,
        num_heads=4,
        qkv_bias=False,
        proj_bias=False,
        with_ffn=False,
        concat_cross_attn=False,
        concat_output=False,
        no_cross_attn=False,
        with_norm=False,
        concat_conv3x3=False,
        ):
        super().__init__()

        out_dim = out_dim or in_dim1

        self.no_cross_attn = no_cross_attn
        self.with_norm = with_norm

        if no_cross_attn:
            if concat_conv3x3:
                self.proj = nn.Conv2d(in_dim1 + in_dim2, out_dim, 3, 1, 1)
            else:
                self.proj = nn.Conv2d(in_dim1 + in_dim2, out_dim, 1)
        else:
            self.with_ffn = with_ffn
            self.concat_cross_attn = concat_cross_attn
            self.concat_output = concat_output
            
            self.cross_attn = CrossAttention(
                in_dim1=in_dim1,
                in_dim2=in_dim2,
                dim=dim,
                out_dim=out_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
            )

            if with_norm:
                self.norm1 = nn.LayerNorm(out_dim)
            else:
                self.norm1 = nn.Identity()

            if with_ffn:
                in_channels = out_dim + in_dim1 if concat_cross_attn else in_dim1
                ffn_dim_expansion = 4
                self.mlp = nn.Sequential(
                    nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                    nn.GELU(),
                    nn.Linear(in_channels * ffn_dim_expansion, in_dim1, bias=False),
                )

                if with_norm:
                    self.norm2 = nn.LayerNorm(in_dim1)
                else:
                    self.norm2 = nn.Identity()

            if self.concat_output:
                self.out = nn.Linear(out_dim + in_dim1, in_dim1)

    def forward(self, x, y):
        # x: [B, C, H, W]
        # y: [B, N, C] or [B, C, H, W]

        if self.no_cross_attn:
            assert x.dim() == 4 and y.dim() == 4
            if y.shape[2:] != x.shape[2:]:
                y = F.interpolate(y, x.shape[2:], mode='bilinear', align_corners=True)
            return self.proj(torch.cat((x, y), dim=1))

        identity = x

        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)

        cross_attn = self.norm1(self.cross_attn(x, y))

        if self.with_ffn:
            if self.concat_cross_attn:
                concat = torch.cat((x, cross_attn), dim=-1)
            else:
                concat = x + cross_attn

            cross_attn = self.norm2(self.mlp(concat))

        if self.concat_output:
            return self.out(torch.cat((x, cross_attn), dim=-1))

        # reshape back
        cross_attn = cross_attn.view(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]

        return identity + cross_attn

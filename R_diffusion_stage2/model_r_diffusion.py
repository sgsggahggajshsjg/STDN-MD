import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.checkpoint as checkpoint
from einops import rearrange


# === Mamba Components (SS2D GRU) ===
class SS2D_PyTorch(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, bias=True, padding=(d_conv - 1) // 2)
        self.act = nn.SiLU()
        # GRU 扫描器
        self.gru_h_f = nn.GRU(self.d_inner, self.d_inner, batch_first=False)
        self.gru_h_b = nn.GRU(self.d_inner, self.d_inner, batch_first=False)
        self.gru_v_f = nn.GRU(self.d_inner, self.d_inner, batch_first=False)
        self.gru_v_b = nn.GRU(self.d_inner, self.d_inner, batch_first=False)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        B, H, W, C_in = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.conv2d(x))
        x = x.permute(0, 2, 3, 1)

        # 向量化扫描
        x_h = x.permute(2, 0, 1, 3).reshape(W, B * H, -1)
        y_h_f, _ = self.gru_h_f(x_h)
        y_h_b, _ = self.gru_h_b(x_h.flip(0))
        y_h_b = y_h_b.flip(0)
        y_h_f = y_h_f.view(W, B, H, -1).permute(1, 2, 0, 3)
        y_h_b = y_h_b.view(W, B, H, -1).permute(1, 2, 0, 3)

        x_v = x.permute(1, 0, 2, 3).reshape(H, B * W, -1)
        y_v_f, _ = self.gru_v_f(x_v)
        y_v_b, _ = self.gru_v_b(x_v.flip(0))
        y_v_b = y_v_b.flip(0)
        y_v_f = y_v_f.view(H, B, W, -1).permute(1, 0, 2, 3)
        y_v_b = y_v_b.view(H, B, W, -1).permute(1, 0, 2, 3)

        out = y_h_f + y_h_b + y_v_f + y_v_b
        out = out * F.silu(z)
        out = self.out_norm(out)
        out = self.out_proj(out)
        return self.dropout(out)


class VSSBlock(nn.Module):
    def __init__(self, dim, use_checkpoint=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.self_attention = SS2D_PyTorch(d_model=dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, input):
        def _forward_impl(x_in):
            x = x_in.permute(0, 2, 3, 1)
            x = x + self.self_attention(self.ln_1(x))
            return x.permute(0, 3, 1, 2)

        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(_forward_impl, input, use_reentrant=False)
        else:
            return _forward_impl(input)



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim // 4, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.mlp(emb)


class ResConvBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.time_proj = nn.Linear(time_emb_dim, out_c)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


class DownSample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, 2, 1)

    def forward(self, x): return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(dim_in, dim_out, 3, 1, 1)

    def forward(self, x): return self.conv(self.up(x))


# === Main Model ===
class HybridMambaUNet(nn.Module):
    def __init__(self, in_channels=8, base_dim=64, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        self.time_dim = base_dim * 4
        self.time_mlp = TimeEmbedding(self.time_dim)
        self.inc = nn.Conv2d(in_channels, base_dim, 3, 1, 1)

        dims = [base_dim, *map(lambda m: base_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResConvBlock(dim_in, dim_in, self.time_dim),
                ResConvBlock(dim_in, dim_in, self.time_dim),
                DownSample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, 1, 1)
            ]))

        mid_dim = dims[-1]
        self.mid1 = VSSBlock(mid_dim, use_checkpoint=True)
        self.mid2 = VSSBlock(mid_dim, use_checkpoint=True)

        self.ups = nn.ModuleList([])
        reversed_dims = list(reversed(dims))
        reversed_in_out = list(zip(reversed_dims[:-1], reversed_dims[1:]))
        for ind, (dim_in, dim_out) in enumerate(reversed_in_out):
            self.ups.append(nn.ModuleList([
                UpSample(dim_in, dim_out),
                ResConvBlock(dim_out * 2, dim_out, self.time_dim),
                ResConvBlock(dim_out, dim_out, self.time_dim),
            ]))

        self.outc = nn.Sequential(
            nn.GroupNorm(8, base_dim), nn.SiLU(), nn.Conv2d(base_dim, 3, 3, 1, 1)
        )

    def forward(self, x, t, i_low, mask, l_gen):
        inp = torch.cat([x, i_low, mask, l_gen], dim=1)
        t_emb = self.time_mlp(t)
        h = self.inc(inp)
        hls = []
        for block1, block2, downsample in self.downs:
            h = block1(h, t_emb)
            h = block2(h, t_emb)
            hls.append(h)
            h = downsample(h)

        h = self.mid1(h)
        h = self.mid2(h)

        for upsample, block1, block2 in self.ups:
            h = upsample(h)
            h_skip = hls.pop()
            if h.shape[2:] != h_skip.shape[2:]:
                h = F.interpolate(h, size=h_skip.shape[2:], mode='nearest')
            h = torch.cat((h, h_skip), dim=1)
            h = block1(h, t_emb)
            h = block2(h, t_emb)

        return self.outc(h)
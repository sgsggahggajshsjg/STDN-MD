import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


class SemanticGuidedAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SemanticGuidedAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.semantic_proj = nn.Sequential(nn.Conv2d(1, dim, kernel_size=1), nn.Sigmoid())

    def forward(self, x, mask):
        b, c, h, w = x.shape
        if mask.shape[-1] != w:
            mask_resized = F.interpolate(mask, size=(h, w), mode='nearest')
        else:
            mask_resized = mask
        semantic_map = self.semantic_proj(mask_resized)
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        s = rearrange(semantic_map, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v) * (1 + s)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = SemanticGuidedAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x



class STDN(nn.Module):
    def __init__(self, dim=48):
        super(STDN, self).__init__()
        inp_channels = 4  # img(3) + mask(1)
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, 1, 1)

        self.enc1 = nn.ModuleList([TransformerBlock(dim, 1, 2.66, False) for _ in range(2)])
        self.down1 = nn.Conv2d(dim, dim * 2, 3, 2, 1)
        self.enc2 = nn.ModuleList([TransformerBlock(dim * 2, 2, 2.66, False) for _ in range(2)])
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 3, 2, 1)
        self.enc3 = nn.ModuleList([TransformerBlock(dim * 4, 4, 2.66, False) for _ in range(4)])

        self.up2 = nn.Sequential(nn.Conv2d(dim * 4, dim * 8, 1), nn.PixelShuffle(2))
        self.dec2 = nn.ModuleList([TransformerBlock(dim * 2, 2, 2.66, False) for _ in range(2)])
        self.up1 = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, 1), nn.PixelShuffle(2))
        self.dec1 = nn.ModuleList([TransformerBlock(dim, 1, 2.66, False) for _ in range(2)])

    def forward(self, img, mask):
        input_tensor = torch.cat([img, mask], dim=1)
        x = self.patch_embed(input_tensor)

        skip1 = x
        for blk in self.enc1: x = blk(x, mask)
        x = self.down1(x)

        skip2 = x
        for blk in self.enc2: x = blk(x, mask)
        x = self.down2(x)

        for blk in self.enc3: x = blk(x, mask)

        x = self.up2(x)
        x = x + skip2
        for blk in self.dec2: x = blk(x, mask)

        x = self.up1(x)
        x = x + skip1
        for blk in self.dec1: x = blk(x, mask)

        # 直接返回特征图 [B, 48, H, W]
        return x
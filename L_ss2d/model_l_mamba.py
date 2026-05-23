import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange


class SS2D_PyTorch(nn.Module):


    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.):
        super().__init__()
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, bias=True, padding=(d_conv - 1) // 2)
        self.act = nn.SiLU()


        self.gru_h_f = nn.GRU(self.d_inner, self.d_inner, batch_first=False)  # 水平前向
        self.gru_h_b = nn.GRU(self.d_inner, self.d_inner, batch_first=False)  # 水平后向
        self.gru_v_f = nn.GRU(self.d_inner, self.d_inner, batch_first=False)  # 垂直前向
        self.gru_v_b = nn.GRU(self.d_inner, self.d_inner, batch_first=False)  # 垂直后向

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


class L_Mamba(nn.Module):
    def __init__(self, stdn_dim=48, hidden_dim=64, num_blocks=4):
        super(L_Mamba, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(stdn_dim + 1, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )


        self.body = nn.Sequential(
            *[VSSBlock(hidden_dim, use_checkpoint=False) for _ in range(num_blocks)]
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, stdn_feat, mask):
        feat_size = stdn_feat.shape[2:]
        mask_resized = F.interpolate(mask, size=feat_size, mode='nearest')
        combined_input = torch.cat([stdn_feat, mask_resized], dim=1)
        fused_feat = self.fusion_conv(combined_input)
        body_feat = self.body(fused_feat)
        l_low_res = self.output_conv(body_feat)
        original_size = mask.shape[2:]
        l_high_res = F.interpolate(l_low_res, size=original_size, mode='bilinear', align_corners=False)
        return l_high_res


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = L_Mamba().to(device)
    x = torch.randn(2, 48, 128, 128).to(device)
    m = torch.randn(2, 1, 512, 512).to(device)
    print("Testing Corrected L-Mamba...")
    out = model(x, m)
    print("Output shape:", out.shape)
    print("✅ Done!")
import torch
import torch.nn as nn
import torch.nn.functional as F


class LMambaGuidedLoss(nn.Module):
    def __init__(self, device,
                 lambda_fid=10.0,
                 lambda_smooth=3.0,
                 lambda_struct=0.0,
                 lambda_exp=0.1,
                 target_exp=0.6):
        super().__init__()
        self.device = device
        self.lambda_fid = lambda_fid
        self.lambda_smooth = lambda_smooth
        self.lambda_struct = lambda_struct
        self.lambda_exp = lambda_exp
        self.target_exp = target_exp


        self.kernel_h = torch.tensor([[[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]]]).float().to(device) / 3.0
        self.kernel_v = torch.tensor([[[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]]).float().to(device) / 3.0

    def get_gradient(self, img):


        if img.shape[1] != 1:

            img = torch.mean(img, dim=1, keepdim=True)

        img_pad = F.pad(img, (1, 1, 1, 1), mode='replicate')
        grad_h = F.conv2d(img_pad, self.kernel_h)
        grad_v = F.conv2d(img_pad, self.kernel_v)
        return grad_h, grad_v

    def get_ref_illumination(self, img_high):

        if img_high.min() < 0:
            img_high = (img_high + 1) / 2.0
        l_ref, _ = torch.max(img_high, dim=1, keepdim=True)
        return l_ref

    def forward(self, l_pred, i_high, mask):

        l_ref = self.get_ref_illumination(i_high)


        loss_fid = F.l1_loss(l_pred, l_ref)


        grad_h_pred, grad_v_pred = self.get_gradient(l_pred)
        grad_mag_pred = torch.abs(grad_h_pred) + torch.abs(grad_v_pred)


        i_high_01 = (i_high + 1) / 2.0
        i_gray = torch.mean(i_high_01, dim=1, keepdim=True)  # [B, 1, H, W]

        grad_h_ref, grad_v_ref = self.get_gradient(i_gray)
        grad_mag_ref = torch.abs(grad_h_ref) + torch.abs(grad_v_ref)


        edge_weight = torch.exp(-10 * grad_mag_ref)


        mask_weight = 1.0 - mask


        final_weight = torch.max(mask_weight, edge_weight)

        loss_smooth = torch.mean(final_weight * grad_mag_pred)

        l_avg = F.avg_pool2d(l_pred, kernel_size=16, stride=16)
        loss_exp = F.l1_loss(l_avg, torch.ones_like(l_avg) * self.target_exp)


        total_loss = (self.lambda_fid * loss_fid +
                      self.lambda_smooth * loss_smooth +
                      self.lambda_exp * loss_exp)

        return total_loss, {
            "fid": loss_fid.item(),
            "struct": 0.0,
            "smooth": loss_smooth.item(),
            "exp": loss_exp.item()
        }
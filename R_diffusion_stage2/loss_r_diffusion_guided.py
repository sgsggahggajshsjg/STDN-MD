import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class RDiffusionGuidedLoss(nn.Module):
    def __init__(self, device,
                 lambda_diff=1.0,
                 lambda_color=2.0,
                 lambda_lpips=1.0,  # LPIPS
                 lambda_rec=1.0):  #
        super().__init__()
        self.device = device
        self.weights = {'diff': lambda_diff, 'color': lambda_color,
                        'lpips': lambda_lpips, 'rec': lambda_rec}

        print(f"Initializing LPIPS (VGG)... Color Weight: {lambda_color}")
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        self.lpips_fn.eval()

    def get_face_mean_color(self, img, mask):

        # img: [-1, 1] -> [0, 1]
        img_01 = (img + 1) / 2.0


        face_region = (mask > 0.8).float()

        num_pixels = face_region.sum(dim=[2, 3], keepdim=True) + 1e-8
        region_sum = (img_01 * face_region).sum(dim=[2, 3], keepdim=True)

        return region_sum / num_pixels

    def forward(self, noise_pred, noise_gt, pred_x0, i_gt, l_high, mask):

        loss_diff = torch.mean(mask * (noise_pred - noise_gt) ** 2)

        i_pred_01 = (pred_x0 + 1) / 2.0
        i_gt_01 = (i_gt + 1) / 2.0
        loss_rec = torch.mean(mask * torch.abs(i_pred_01 - i_gt_01))


        mean_pred = self.get_face_mean_color(pred_x0, mask)
        mean_gt = self.get_face_mean_color(i_gt, mask)
        loss_color = F.l1_loss(mean_pred, mean_gt)


        lpips_val = self.lpips_fn(pred_x0, i_gt)


        if lpips_val.shape[-1] > 1:
            loss_lpips = torch.mean(mask * lpips_val)
        else:
            loss_lpips = lpips_val.mean()

        total = (self.weights['diff'] * loss_diff +
                 self.weights['color'] * loss_color +
                 self.weights['lpips'] * loss_lpips +
                 self.weights['rec'] * loss_rec)

        return total, {
            "diff": loss_diff.item(),
            "color": loss_color.item(),
            "lpips": loss_lpips.item(),
            "rec": loss_rec.item()
        }
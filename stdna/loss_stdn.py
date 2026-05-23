import torch
import torch.nn as nn
import torch.nn.functional as F


class DecomLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2))
        self.kernel_y = torch.FloatTensor([[0, -1], [0, 1]]).view((1, 1, 2, 2))

    def gradient(self, input_tensor, direction):

        b, c, h, w = input_tensor.shape


        if direction == "x":
            kernel = self.kernel_x.to(input_tensor.device)
        else:
            kernel = self.kernel_y.to(input_tensor.device)


        kernel = kernel.repeat(c, 1, 1, 1)


        grad_out = torch.abs(F.conv2d(input_tensor, kernel, padding=1, groups=c))


        return grad_out[:, :, :h, :w]

    def forward(self, R, L, I_input, Mask):

        recon_weight = 1 + 9 * Mask

        loss_recon = torch.mean(torch.abs(R * L - I_input) * recon_weight)


        grad_L_x = self.gradient(L, "x")
        grad_L_y = self.gradient(L, "y")

        grad_I_x = self.gradient(I_input, "x")
        grad_I_y = self.gradient(I_input, "y")


        grad_I_x_mean = torch.mean(grad_I_x, dim=1, keepdim=True)
        grad_I_y_mean = torch.mean(grad_I_y, dim=1, keepdim=True)


        weight_x = torch.exp(-10 * grad_I_x_mean) * (1 + 5 * Mask)
        weight_y = torch.exp(-10 * grad_I_y_mean) * (1 + 5 * Mask)

        loss_smooth = torch.mean(grad_L_x * weight_x) + torch.mean(grad_L_y * weight_y)


        loss_R_const = torch.mean(torch.abs(R - I_input) * Mask)


        total_loss = loss_recon + 0.1 * loss_smooth + 0.01 * loss_R_const

        return total_loss
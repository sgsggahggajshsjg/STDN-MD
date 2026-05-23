import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import datetime

# 导入模块
from dataset_stage2 import Stage2Dataset
from model_r_diffusion import HybridMambaUNet
from loss_r_diffusion_guided import RDiffusionGuidedLoss

# ================= 配置区域 =================
PROJECT_ROOT = '/home/ubuntu/zpc/lunwen111/'
DATA_ROOT = os.path.join(PROJECT_ROOT, 'dataset')

SAVE_DIR = os.path.join(PROJECT_ROOT, 'experiments/stage2_direct_generation_7/')
SAMPLE_DIR = os.path.join(SAVE_DIR, 'samples')


RESUME = False
RESUME_WEIGHT_PATH = os.path.join(SAVE_DIR, 'r_diffusion_best.pth')

IMG_SIZE = 256
BATCH_SIZE = 16
LR = 1e-4
MIN_LR = 1e-6
EPOCHS = 150
TIMESTEPS = 1000
VAL_INTERVAL = 5
DEVICE = 'cuda'




class DDPMScheduler:
    def __init__(self, timesteps=1000, device=DEVICE):
        self.timesteps = timesteps
        self.device = device
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def add_noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    def predict_x0(self, x_t, t, noise_pred):
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_0 = (x_t - sqrt_one_minus * noise_pred) / (sqrt_alpha + 1e-5)
        return torch.clamp(x_0, -1., 1.)

    @torch.no_grad()
    def sample(self, model, shape, cond_i, cond_mask, cond_l):
        b = shape[0]
        img = torch.randn(shape, device=self.device)
        iterator = reversed(range(0, self.timesteps))
        for i in iterator:
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            noise_pred = model(img, t, cond_i, cond_mask, cond_l)
            mean = self.sqrt_recip_alphas[i] * (
                    img - self.betas[i] / self.sqrt_one_minus_alphas_cumprod[i] * noise_pred
            )
            if i > 0:
                noise = torch.randn_like(img)
                sigma = torch.sqrt(self.posterior_variance[i])
                img = mean + sigma * noise
            else:
                img = mean
        return torch.clamp(img, -1., 1.)


def train():
    os.makedirs(SAVE_DIR + 'weights', exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = open(os.path.join(SAVE_DIR, f'train_log_{current_time}.txt'), 'w')

    train_dataset = Stage2Dataset(DATA_ROOT, 'train', img_size=IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    try:
        val_dataset = Stage2Dataset(DATA_ROOT, 'test', img_size=IMG_SIZE)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        fixed_val_batch = next(iter(val_loader))
        print("✅ Fixed validation batch loaded from 'test' set.")
    except Exception as e:
        print(f"⚠️ Could not load test set ({e}), using train set instead.")
        val_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        fixed_val_batch = next(iter(val_loader))

    val_low = fixed_val_batch['low'].to(DEVICE)
    val_gt = fixed_val_batch['r_gt'].to(DEVICE)
    val_mask = fixed_val_batch['mask'].to(DEVICE)
    val_l = fixed_val_batch['l_gen'].to(DEVICE)

    model = HybridMambaUNet(in_channels=8, base_dim=64).to(DEVICE)

    if RESUME and os.path.exists(RESUME_WEIGHT_PATH):
        print(f"🔄 Resuming from: {RESUME_WEIGHT_PATH}")
        model.load_state_dict(torch.load(RESUME_WEIGHT_PATH, map_location=DEVICE))
    else:
        print("🚀 Starting training from scratch (Recommended).")


    loss_func = RDiffusionGuidedLoss(
        DEVICE,
        lambda_diff=1.0,
        lambda_color=2.0,
        lambda_lpips=1.0,
        lambda_rec=1.0
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)


    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MIN_LR
    )

    scheduler_ddpm = DDPMScheduler(TIMESTEPS, DEVICE)

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        stats = {"diff": 0, "color": 0, "lpips": 0, "rec": 0}


        current_lr = optimizer.param_groups[0]['lr']

        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}/{EPOCHS} [LR={current_lr:.2e}]")
        for batch in loop:
            i_low = batch['low'].to(DEVICE)
            i_gt = batch['r_gt'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            l_gen = batch['l_gen'].to(DEVICE)

            t = torch.randint(0, TIMESTEPS, (i_low.shape[0],), device=DEVICE).long()
            x_t, noise_gt = scheduler_ddpm.add_noise(i_gt, t)

            optimizer.zero_grad()
            noise_pred = model(x_t, t, i_low, mask, l_gen)
            i_pred_x0 = scheduler_ddpm.predict_x0(x_t, t, noise_pred)

            loss, loss_dict = loss_func(noise_pred, noise_gt, i_pred_x0, i_gt, l_gen, mask)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            for k, v in loss_dict.items(): stats[k] += v
            loop.set_postfix(loss=f"{loss.item():.4f}", col=f"{loss_dict['color']:.4f}")


        scheduler_lr.step()

        avg_loss = epoch_loss / len(train_loader)

        msg = f"Ep {epoch + 1}: Avg Loss: {avg_loss:.5f} | Color: {stats['color'] / len(train_loader):.4f} | LR: {current_lr:.2e}"
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'r_diffusion_best.pth'))

        if (epoch + 1) % VAL_INTERVAL == 0:
            print(f"🎨 Sampling preview for Epoch {epoch + 1}...")
            model.eval()
            with torch.no_grad():
                target_shape = val_low.shape
                val_l_clamped = torch.clamp(val_l, 0.0, 1.0)
                generated = scheduler_ddpm.sample(model, target_shape, val_low, val_mask, val_l_clamped)

                vis_low = (val_low + 1) / 2.0
                vis_gt = (val_gt + 1) / 2.0
                vis_gen = (generated + 1) / 2.0
                vis_l = val_l_clamped.repeat(1, 3, 1, 1)

                grid = torch.cat([vis_low, vis_l, vis_gen, vis_gt], dim=3)
                save_path = os.path.join(SAMPLE_DIR, f"epoch_{epoch + 1}.png")
                save_image(grid, save_path, nrow=1, normalize=False)

            print(f"🖼️ Preview saved to: {save_path}")
            model.train()

    log_file.close()


if __name__ == '__main__':
    train()
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

# 导入自定义模块
from dataset_stage1_guided import GuidedLowLightDataset
from stdna.model_stdn import STDN
from model_l_mamba import L_Mamba
from loss_stage1_guided import LMambaGuidedLoss

# ================= 配置区域 =================
PROJECT_ROOT = '/home/ubuntu/zpc/lunwen111/'
DATA_ROOT = os.path.join(PROJECT_ROOT, 'dataset')
SAVE_DIR = os.path.join(PROJECT_ROOT, 'experiments/stage1_guided_blur_3/')


RESUME = True

RESUME_STDN_PATH = os.path.join(PROJECT_ROOT, 'experiments/stage1_guided_blur_2/stdn_best.pth')
RESUME_LMAMBA_PATH = os.path.join(PROJECT_ROOT, 'experiments/stage1_guided_blur_2/l_mamba_best.pth')


IMG_SIZE = 128
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 100
NUM_WORKERS = 8

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Training on device: {DEVICE}")


# ===========================================

def train():
    os.makedirs(SAVE_DIR + 'weights', exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = open(os.path.join(SAVE_DIR, f'train_log_resume_{current_time}.txt'), 'w')


    train_dataset = GuidedLowLightDataset(
        low_dir=os.path.join(DATA_ROOT, 'train/low'),
        high_dir=os.path.join(DATA_ROOT, 'train/high'),
        mask_dir=os.path.join(DATA_ROOT, 'train/mask'),
        img_size=IMG_SIZE, is_train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)


    stdn_model = STDN(dim=48).to(DEVICE)
    l_mamba_model = L_Mamba(stdn_dim=48).to(DEVICE)


    if RESUME:
        if os.path.exists(RESUME_STDN_PATH) and os.path.exists(RESUME_LMAMBA_PATH):
            print(f"🔄 Resuming training from: {RESUME_STDN_PATH}")
            stdn_model.load_state_dict(torch.load(RESUME_STDN_PATH, map_location=DEVICE))
            l_mamba_model.load_state_dict(torch.load(RESUME_LMAMBA_PATH, map_location=DEVICE))
            print("✅ Checkpoints loaded successfully!")
        else:
            print("⚠️ Warning: Checkpoint files not found! Starting from scratch.")


    loss_func = LMambaGuidedLoss(
        device=DEVICE,
        lambda_fid=10.0,
        lambda_smooth=3.0,
        lambda_struct=0.0,
        lambda_exp=0.1,
        target_exp=0.6
    ).to(DEVICE)


    optimizer = optim.AdamW(
        list(stdn_model.parameters()) + list(l_mamba_model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)


    best_loss = float('inf')

    print("🚀 Start Resuming Training...")

    for epoch in range(EPOCHS):
        stdn_model.train()
        l_mamba_model.train()

        epoch_loss = 0
        loss_stats = {"fid": 0, "struct": 0, "smooth": 0, "exp": 0}
        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}/{EPOCHS}")

        for batch in loop:
            low_img = batch['low'].to(DEVICE)
            high_img = batch['high'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)

            optimizer.zero_grad()

            stdn_feat = stdn_model(low_img, mask)
            pred_l = l_mamba_model(stdn_feat, mask)


            total_loss, batch_loss_stats = loss_func(pred_l, high_img, mask)


            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(stdn_model.parameters()) + list(l_mamba_model.parameters()), max_norm=1.0
            )
            optimizer.step()

            epoch_loss += total_loss.item()
            for k, v in batch_loss_stats.items(): loss_stats[k] += v


            loop.set_postfix(loss=f"{total_loss.item():.4f}", smooth=f"{batch_loss_stats['smooth']:.4f}")

        scheduler.step()


        avg_epoch_loss = epoch_loss / len(train_loader)
        for k in loss_stats: loss_stats[k] /= len(train_loader)

        log_msg = (f"Epoch [{epoch + 1}/{EPOCHS}] Avg Loss: {avg_epoch_loss:.6f} | "
                   f"Fid: {loss_stats['fid']:.4f} | Struct: {loss_stats['struct']:.4f} | "
                   f"Smooth: {loss_stats['smooth']:.4f} | Exp: {loss_stats['exp']:.4f}")
        print(log_msg)
        log_file.write(log_msg + '\n')
        log_file.flush()


        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

            torch.save(stdn_model.state_dict(), os.path.join(SAVE_DIR, 'stdn_best.pth'))
            torch.save(l_mamba_model.state_dict(), os.path.join(SAVE_DIR, 'l_mamba_best.pth'))
            print("✨ New best model saved!")

    print("🎉 Training Finished!")


if __name__ == '__main__':
    train()
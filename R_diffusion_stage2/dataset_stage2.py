import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


class Stage2Dataset(Dataset):
    def __init__(self, data_root, subset='train', img_size=256):
        self.img_size = img_size

        # 定义路径
        self.low_dir = os.path.join(data_root, subset, 'low')
        self.high_gt_dir = os.path.join(data_root, subset, 'high')
        self.mask_dir = os.path.join(data_root, subset, 'mask')

        self.l_gen_dir = os.path.join(data_root, subset, 'l_high_gen_blur_3')


        IMG_EXTS = ('.jpg', '.jpeg', '.png')

        self.file_pairs = []
        # 扫描文件
        if os.path.exists(self.low_dir):
            low_files = sorted([f for f in os.listdir(self.low_dir) if f.lower().endswith(IMG_EXTS)])
            print(f"Scanning Stage 2 data in {self.low_dir}...")

            for low_f in low_files:
                basename = os.path.splitext(low_f)[0]


                high_f = None
                for ext in IMG_EXTS:
                    if os.path.exists(os.path.join(self.high_gt_dir, basename + ext)):
                        high_f = basename + ext
                        break


                mask_f = basename + '.png'
                l_gen_f = basename + '.png'

                if high_f and \
                        os.path.exists(os.path.join(self.mask_dir, mask_f)) and \
                        os.path.exists(os.path.join(self.l_gen_dir, l_gen_f)):
                    self.file_pairs.append({
                        'low': os.path.join(self.low_dir, low_f),
                        'high': os.path.join(self.high_gt_dir, high_f),
                        'mask': os.path.join(self.mask_dir, mask_f),
                        'l_gen': os.path.join(self.l_gen_dir, l_gen_f)
                    })

        print(f"[{subset}] Found {len(self.file_pairs)} valid quadruplets.")


        self.norm_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])


        self.l_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),


            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            # ============================

            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        paths = self.file_pairs[idx]

        low = Image.open(paths['low']).convert('RGB')
        high = Image.open(paths['high']).convert('RGB')
        mask_pil = Image.open(paths['mask'])
        l_gen = Image.open(paths['l_gen'])


        t_low = self.norm_transform(low)
        t_high = self.norm_transform(high)
        t_l_gen = self.l_transform(l_gen)


        raw_mask = self.mask_transform(mask_pil)


        balanced_mask = torch.zeros_like(raw_mask)

        balanced_mask[(raw_mask > 0.05) & (raw_mask < 0.2)] = 0.5
        balanced_mask[(raw_mask > 0.4) & (raw_mask < 0.6)] = 0.9
        balanced_mask[(raw_mask > 0.9)] = 1.4

        return {
            'low': t_low,
            'r_gt': t_high,
            'mask': balanced_mask,
            'l_gen': t_l_gen
        }
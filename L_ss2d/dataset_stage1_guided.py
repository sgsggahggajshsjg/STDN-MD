# dataset_stage1_guided.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


class GuidedLowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir, mask_dir, img_size=512, is_train=True):

        self.low_dir = low_dir
        self.high_dir = high_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train


        self.file_pairs = []


        low_files = sorted([f for f in os.listdir(low_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        print(f"Scanning files in {low_dir}...")
        for low_filename in low_files:

            basename = os.path.splitext(low_filename)[0]


            high_filename = low_filename
            mask_filename = basename + '.png'

            high_path = os.path.join(self.high_dir, high_filename)
            mask_path = os.path.join(self.mask_dir, mask_filename)


            if os.path.exists(high_path) and os.path.exists(mask_path):

                self.file_pairs.append({
                    'basename': basename,
                    'low_file': low_filename,
                    'high_file': high_filename,
                    'mask_file': mask_filename
                })
            else:

                pass

        stage_name = "Training" if is_train else "Validation"
        print(f"[{stage_name}] Found {len(self.file_pairs)} valid image pairs (Low+High+Mask).")


        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


        self.transform_mask = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):

        pair_info = self.file_pairs[idx]


        low_path = os.path.join(self.low_dir, pair_info['low_file'])
        high_path = os.path.join(self.high_dir, pair_info['high_file'])
        mask_path = os.path.join(self.mask_dir, pair_info['mask_file'])

        try:
            low_img = Image.open(low_path).convert('RGB')
            high_img = Image.open(high_path).convert('RGB')
            mask_img = Image.open(mask_path)


            low_tensor = self.transform_img(low_img)
            high_tensor = self.transform_img(high_img)
            mask_tensor = self.transform_mask(mask_img)

            return {
                'low': low_tensor,
                'high': high_tensor,
                'mask': mask_tensor,
                'filename': pair_info['basename']
            }
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            print(f"Paths: {low_path}, {high_path}, {mask_path}")

            raise e


if __name__ == '__main__':

    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    import shutil

    print("--- Starting Dataset Test ---")


    test_root = './temp_test_dataset'
    low_dir = os.path.join(test_root, 'low')
    high_dir = os.path.join(test_root, 'high')
    mask_dir = os.path.join(test_root, 'mask')
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)


    Image.new('RGB', (100, 100), color=(50, 50, 50)).save(os.path.join(low_dir, 'pair1.jpg'))
    Image.new('RGB', (100, 100), color=(200, 200, 200)).save(os.path.join(high_dir, 'pair1.jpg'))
    Image.new('L', (100, 100), color=128).save(os.path.join(mask_dir, 'pair1.png'))

    Image.new('RGB', (100, 100)).save(os.path.join(low_dir, 'unpaired.png'))


    try:
        dataset = GuidedLowLightDataset(
            low_dir=low_dir,
            high_dir=high_dir,
            mask_dir=mask_dir,
            img_size=256,
            is_train=False
        )
    except Exception as e:
        print(f"Dataset initialization failed: {e}")
        exit(1)


    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        print("\n--- Batch Check ---")

        for batch in loader:
            low = batch['low']
            high = batch['high']
            mask = batch['mask']
            filenames = batch['filename']

            print(f"Filenames in batch: {filenames}")
            print(f"Low shape: {low.shape}, Range: [{low.min():.3f}, {low.max():.3f}]")
            print(f"High shape: {high.shape}, Range: [{high.min():.3f}, {high.max():.3f}]")
            # 重点检查 Mask 的值是否正确
            print(f"Mask shape: {mask.shape}, Unique values (approx): {torch.unique(mask)}")

            # 简单的断言检查
            assert low.shape == high.shape
            assert mask.shape[1] == 1
            assert low.min() >= -1.0 and low.max() <= 1.0
            assert mask.min() >= 0.0 and mask.max() <= 1.0
            print("✅ Basic assertions passed!")

            break
    else:
        print("❌ No valid pairs found for testing.")


    try:
        shutil.rmtree(test_root)
        print("\nTemp test directory cleaned up.")
    except OSError as e:
        print(f"Error cleaning up {test_root}: {e}")

    print("--- Test finished ---")
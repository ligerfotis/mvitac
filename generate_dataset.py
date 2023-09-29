from __future__ import print_function

from PIL import Image

import torch
import os
from pathlib import Path
from torch.utils.data import Dataset


class TouchFolderLabel(Dataset):
    """Folder datasets which returns the index of the image as well."""

    def __init__(self, root, transform=None, target_transform=None, two_crop=False,
                 mode='train', label='full', data_amount=100):
        self.two_crop = two_crop
        self.dataroot = Path('/home/vedant/dataset/')
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.label = label

        # Read appropriate data based on mode and label
        if label == 'rough' and mode in ['train', 'test']:
            data_file = os.path.join(root, f'{mode}_rough.txt')
        else:
            data_file = os.path.join(root, f'{mode}.txt')

        with open(data_file, 'r') as f:
            self.env = [line.strip() for line in f.readlines()]

        self.length = len(self.env)

    def __getitem__(self, index):
        """Returns the image, target, and index."""
        assert index < self.length, 'index_A range error'

        raw, target = self.env[index].split(',')
        target = int(target)

        if self.label == 'hard' and target in [7, 8, 9, 11, 13]:
            target = 1
        elif self.label == 'hard':
            target = 0

        # Construct paths for image and gelsight
        idx = Path(raw).name
        dir_path = self.dataroot / raw[:16]
        A_img_path = dir_path / 'video_frame' / idx
        A_gelsight_path = dir_path / 'gelsight_frame' / idx

        # Load image and gelsight
        A_img = Image.open(A_img_path).convert('RGB')
        A_gel = Image.open(A_gelsight_path).convert('RGB')

        if self.transform:
            A_img = self.transform(A_img)
            A_gel = self.transform(A_gel)

        # out = torch.cat((A_img, A_gel), dim=0)

        # if self.mode == 'pretrain':
        #     return A_img, A_gel, target, index

        return A_img, A_gel, target

    def __len__(self):
        """Return the total number of images."""
        return self.length


class CalandraLabel(Dataset):
    def __init__(self, root_dir, transform=None, train=True, mode='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.subset = "train" if mode == "train" else "test"
        self.samples = []

        # Define the root path based on mode
        subset_path = self.root_dir / self.subset
        modality_path = subset_path / 'gelsightA'

        # Loop over each file in the category and add it to the samples list
        for img_file in modality_path.iterdir():
            if img_file.suffix in [".png"]:
                id = img_file.stem.split('_')[-1]
                phase = img_file.stem.split('_')[-2]
                modality = img_file.stem.split('_')[-3]
                is_success = img_file.stem.split('_')[-4]
                object_name = '_'.join(img_file.stem.split('_')[:-4])

                if self.subset == "test" and phase != "during":
                    continue

            # Construct the paths for each modality
            paths = {
                'gelsightA': subset_path / 'gelsightA' / img_file.name,
                'gelsightB': subset_path / 'gelsightB' / f"{object_name}_{is_success}_gelsightB_{phase}_{id}.png",
                'kinectA_rgb': subset_path / 'kinectA_rgb' / f"{object_name}_{is_success}_kinectA_rgb_{phase}_{id}.png"
            }

            # Check if all files exist
            if all(p.exists() for p in paths.values()):
                self.samples.append((paths['gelsightA'], paths['gelsightB'], paths['kinectA_rgb']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gelA_path, gelB_path, rgb_path = self.samples[idx]

        # Load images
        gelA_image = Image.open(gelA_path).convert("RGB")
        gelB_image = Image.open(gelB_path).convert("RGB")
        rgb_image = Image.open(rgb_path).convert("RGB")

        # if self.transform:
        gelA_image_q, gelA_image_k = self.transform(gelA_image)
        gelB_image_q, gelB_image_k = self.transform(gelB_image)
        rgb_image_q, rgb_image_k = self.transform(rgb_image)

        stacked_gelsight_images_q = torch.cat((gelA_image_q, gelB_image_q), dim=0)
        stacked_gelsight_images_k = torch.cat((gelA_image_k, gelB_image_k), dim=0)

        # Extract labels from the file path
        label = torch.tensor(1 if "success" in gelA_path.name else 0, dtype=torch.long)

        # image = torch.cat((stacked_gelsight_images, rgb_image), dim=0)
        return rgb_image_q, rgb_image_k, stacked_gelsight_images_q, stacked_gelsight_images_k, label

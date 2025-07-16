import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch

class PeopleDataset(Dataset):
    def __init__(self, root_path, split="train", transform_size=(512, 512)):
        self.image_dir = os.path.join(root_path, split, "images")
        self.mask_dir = os.path.join(root_path, split, "masks")

        self.image_files = {
            Path(f).stem: os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        }

        self.mask_files = {
            Path(f).stem: os.path.join(self.mask_dir, f)
            for f in os.listdir(self.mask_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        }

        self.keys = sorted(set(self.image_files.keys()) & set(self.mask_files.keys()))

        self.resize_img = transforms.Resize(transform_size)
        self.resize_mask = transforms.Resize(transform_size, interpolation=Image.NEAREST)
        self.normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

    def __getitem__(self, index):
        key = self.keys[index]
        img = Image.open(self.image_files[key]).convert("RGB")
        mask = Image.open(self.mask_files[key]).convert("L")

        img = self.resize_img(img)
        mask = self.resize_mask(mask)

        img = np.asarray(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        img = self.normalize(img)

        mask = np.asarray(mask, dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).contiguous()

        return img, mask

    def __len__(self):
        return len(self.keys)

# --------------------- #
# Verificación sin visualización
# --------------------- #
if __name__ == "__main__":
    dataset = PeopleDataset(root_path="people_data", split="train")
    print(f"Número total de pares válidos: {len(dataset)}")

    img, mask = dataset[0]

    print(f"✔ Imagen tensor creada correctamente:")
    print(f"   Shape: {img.shape}, dtype: {img.dtype}, rango: [{img.min().item():.2f}, {img.max().item():.2f}]")

    print(f"✔ Máscara tensor creada correctamente:")
    print(f"   Shape: {mask.shape}, dtype: {mask.dtype}, valores únicos: {torch.unique(mask)}")

import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch

class PeopleDataset(Dataset):
    def __init__(self, root_path, split="train", transform_size=(512, 512)):
        print("Inicializando dataset...")
        self.image_dir = os.path.join(root_path, split, "images")
        self.mask_dir = os.path.join(root_path, split, "masks")

        # Collect image file paths and store them in a dictionary with their stem as key
        self.image_files = {
            Path(f).stem: os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        }

        # Collect mask file paths and store them in a dictionary with their stem as key
        self.mask_files = {
            Path(f).stem: os.path.join(self.mask_dir, f)
            for f in os.listdir(self.mask_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        }

        # Find common keys (stems) between images and masks to ensure valid pairs
        self.keys = sorted(set(self.image_files.keys()) & set(self.mask_files.keys()))
        print(f"Dataset cargado con {len(self.keys)} pares válidos.")

        # Define image resizing transformation
        self.resize_img = transforms.Resize(transform_size)
        # Define mask resizing transformation, using NEAREST interpolation for masks
        self.resize_mask = transforms.Resize(transform_size, interpolation=Image.NEAREST)
        # Define normalization transformation for images (mean and std for each channel)
        self.normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

    def __getitem__(self, index):
        key = self.keys[index] # Get the key (file stem) for the current index

        img_path = self.image_files[key] # Get the full path for the image
        mask_path = self.mask_files[key] # Get the full path for the mask

        # Open image and convert to RGB format
        img = Image.open(img_path).convert("RGB")
        # Open mask and convert to grayscale (L) format
        mask = Image.open(mask_path).convert("L")

        # Resize image and mask to the specified transform_size
        img = self.resize_img(img)
        mask = self.resize_mask(mask)

        # Convert image to NumPy array, normalize to [0, 1] range, and ensure float32 type
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Convert NumPy image array to PyTorch tensor.
        img_tensor = torch.tensor(img_np).permute(2, 0, 1).contiguous()

        # Normalize the image tensor
        img_tensor = self.normalize(img_tensor)

        # Convert mask to NumPy array, normalize to [0, 1] range, and ensure float32 type
        mask_np = np.array(mask, dtype=np.float32) / 255.0

        # Convert NumPy mask array to PyTorch tensor.
        mask_tensor = torch.tensor(mask_np).unsqueeze(0).contiguous()

        return img_tensor, mask_tensor

    def __len__(self):
        # Return the total number of valid image-mask pairs
        return len(self.keys)

# Test code (only runs when the script is executed directly)
if __name__ == "__main__":
    # Initialize the dataset with a dummy root_path (replace with your actual data path)
    # For testing, ensure you have 'people_data/train/images' and 'people_data/train/masks'
    # directories with some image and mask files.
    dataset = PeopleDataset(root_path="people_data", split="train")

    # Try to access the first item in the dataset
    try:
        img, mask = dataset[0]

        print("\nTensor imagen creado:")
        print(f"   Shape: {img.shape}, dtype: {img.dtype}, rango: [{img.min():.2f}, {img.max():.2f}]")

        print("Tensor máscara creada:")
        print(f"   Shape: {mask.shape}, dtype: {mask.dtype}, valores únicos: {torch.unique(mask)}")
    except IndexError:
        print("\nError: No se pudieron cargar elementos del dataset. Asegúrate de que 'people_data/train/' contiene imágenes y máscaras.")
    except Exception as e:
        print(f"\nSe produjo un error al acceder al primer elemento del dataset: {e}")

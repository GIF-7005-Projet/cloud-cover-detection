import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import tv_tensors


class CloudCoverDataset(Dataset):
    def __init__(
            self,
            X_paths: list[list[Path]],
            y_paths: list[Path],
            transforms=None
    ):
        super().__init__()
        self.X_paths = X_paths
        self.y_paths = y_paths
        self.transforms = transforms
    
    def __len__(self) -> int:
        return len(self.X_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_paths = self.X_paths[index]
        y_path = self.y_paths[index]

        x = tv_tensors.Image(torch.from_numpy(self.load_x(x_paths)))
        y = tv_tensors.Mask(torch.from_numpy(self.load_y(y_path)), dtype=torch.int64)

        if self.transforms:
            sample = self.transforms({'image': x, 'mask': y})
            x, y = sample['image'], sample['mask']
        
        return x, y
        
    def load_x(self, x_paths: list[Path]) -> np.ndarray:
        """
        Inputs: list of path where each path is a feature image
        Outputs: numpy array of shape (channels, height, width)
        """
        return np.array([np.array(Image.open(x_path), dtype=np.int16) for x_path in x_paths])
    
    def load_y(self, y_path: Path) -> np.ndarray:
        """
        Outputs: numpy array of shape (height, width)
        """
        return np.array(Image.open(y_path), dtype=np.int8)

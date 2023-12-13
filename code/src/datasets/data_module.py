import pytorch_lightning as pl
from pathlib import Path
from datasets.imageio import get_X_paths, get_y_paths
from datasets.dataset import CloudCoverDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import Optional
from torchvision.transforms import v2
from datasets.transforms.minmax_normalize import MinMaxNormalize


# TODO Change default mean and std for infrared channel (these are just filler values)
class CloudCoverDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_X_folder_path: Path,
            train_y_folder_path: Path,
            test_X_folder_path: Path,
            test_y_folder_path: Path,
            train_batch_size: int = 32,
            val_batch_size: int = 32,
            test_batch_size: int = 32,
            val_size: float = 0.,
            train_transforms: v2.Compose = v2.Compose([
                MinMaxNormalize(0, 1),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=90),
                v2.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
            ]),
            eval_transforms: v2.Compose = v2.Compose([
                MinMaxNormalize(0, 1),
                v2.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
            ]),
            random_state: int = 42
        ):
        self.train_X_folder_path = train_X_folder_path
        self.train_y_folder_path = train_y_folder_path
        self.test_X_folder_path = test_X_folder_path
        self.test_y_folder_path = test_y_folder_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.val_size = val_size
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms
        self.random_state = random_state
    
    def prepare_data(self):
        self.train_X_paths = get_X_paths(self.train_X_folder_path)
        self.train_y_paths = get_y_paths(self.train_y_folder_path)

        self.test_X_paths = get_X_paths(self.test_X_folder_path)
        self.test_y_paths = get_y_paths(self.test_y_folder_path)
    
    def setup(self, stage: Optional[str]):
        if stage == "fit" or stage is None:

            if self.val_size == 0:
                self.train_dataset = CloudCoverDataset(
                    X_paths=self.train_X_paths, 
                    y_paths=self.train_y_paths,
                    transforms=self.train_transforms
                )
            else:
                train_X_paths_split, val_X_paths_split, train_y_paths_split, val_y_paths_split = train_test_split(self.train_X_paths, self.train_y_paths, test_size=self.val_size, random_state=self.random_state)

                self.train_dataset = CloudCoverDataset(
                    X_paths=train_X_paths_split, 
                    y_paths=train_y_paths_split, 
                    transforms=self.train_transforms
                )

                self.val_dataset = CloudCoverDataset(
                    X_paths=val_X_paths_split, 
                    y_paths=val_y_paths_split, 
                    transforms=self.eval_transforms
                )
        
        if stage == "test":
            self.test_dataset = CloudCoverDataset(
                X_paths=self.test_X_paths, 
                y_paths=self.test_y_paths,
                transforms=self.eval_transforms
            )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)

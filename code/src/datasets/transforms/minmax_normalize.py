import torch
import torch.nn as nn

class MinMaxNormalize(nn.Module):
    def __init__(self, target_min: int, target_max: int):
        super().__init__()
        self.target_min = target_min
        self.target_max = target_max
    
    def forward(self, sample: dict) -> torch.Tensor:
        """
        Inputs: image : shape (channels, height, width) or (batch, channels, height, width).
        Outputs: image : shape (channels, height, width) or (batch, channels, height, width) with values in the range [target_min, target_max].
        """
        img = sample['image']

        is_batched = len(img.shape) == 4

        if not is_batched:
            img = img.unsqueeze(0)
        
        channel_mins = img.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        channel_maxs = img.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

        sample['image'] = ((img - channel_mins) / (channel_maxs - channel_mins)) * (self.target_max - self.target_min) + self.target_min
        
        if not is_batched:
            sample['image'] = sample['image'].squeeze(0)

        return sample

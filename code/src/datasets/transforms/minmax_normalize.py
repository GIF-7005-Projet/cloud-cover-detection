import torch

class MinMaxNormalize():
    def __init__(self, target_min: int, target_max: int):
        self.target_min = target_min
        self.target_max = target_max
    
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Inputs: shape (batch, channels, height, width).
        Outputs: shape (batch, channels, height, width) with values in the range [target_min, target_max].
        """
        channel_mins = sample.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        channel_maxs = sample.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

        return ((sample - channel_mins) / (channel_maxs - channel_mins)) * (self.target_max - self.target_min) + self.target_min

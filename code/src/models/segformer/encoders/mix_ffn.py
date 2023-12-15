import torch
import torch.nn as nn
import torch.nn.functional as F
from models.segformer.encoders.dwconv import DWConv


class MixFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        dropout: float = 0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=hidden_features)
        
        self.conv = DWConv(dim=hidden_features)

        self.fc2 = nn.Linear(in_features=hidden_features,
                             out_features=out_features)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
            H: int
            W: int
        Returns:
            x: (B, N, C)
        """
        x = self.fc1(x)
        x = self.conv(x, H, W)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x

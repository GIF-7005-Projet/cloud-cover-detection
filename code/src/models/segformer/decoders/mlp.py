import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int
    ):
        super().__init__()
        self.fc = nn.Linear(in_features=input_dim,
                            out_features=embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim, H, W)
        Returns:
            x: (B, H * W, embedding_dim)
        """
        x = x.flatten(start_dim=2, end_dim=-1).transpose(1, 2) # (B, H * W, C)
        x = self.fc(x)
        return x

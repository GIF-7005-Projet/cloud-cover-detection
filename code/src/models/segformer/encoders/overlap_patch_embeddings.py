import torch
import torch.nn as nn


class OverlapPatchEmbeddings(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        stride: int,
        in_channels: int = 4,
        embedding_dim: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.embeddings_projection = nn.Conv2d(in_channels=in_channels,
                                               out_channels=embedding_dim,
                                               kernel_size=patch_size,
                                               stride=stride,
                                               padding=(patch_size // 2, patch_size // 2))
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x' : (B, H' * W', embedding_dim)
            num_patches_H: int
            num_patches_W: int
        """
        x = self.embeddings_projection(x) # (B, embedding_dim, H', W')
        
        num_patches_H = x.shape[2]
        num_patches_W = x.shape[3]
        
        x = x.flatten(start_dim=2, end_dim=-1) # (B, embedding_dim, H' * W')
        
        x = self.reshape_for_layer_norm(x) # (B, H' * W', embedding_dim)
        
        x = self.layer_norm(x) # (B, H' * W', embedding_dim)
        
        return x, num_patches_H, num_patches_W

    def reshape_for_layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)

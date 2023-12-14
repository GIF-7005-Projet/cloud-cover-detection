import torch
import torch.nn as nn
import torch.nn.functional as F
from models.segformer.decoders.mlp import MLP
from models.segformer.resize import resize


class SegFormerSemanticSegmentationHead(nn.Module):
    def __init__(
        self,
        c1_in_channels: int,
        c2_in_channels: int,
        c3_in_channels: int,
        c4_in_channels: int,
        num_classes: int,
        embedding_dim: int = 768,
        dropout: float = 0.
    ):
        super().__init__()
        self.linear_c4 = MLP(input_dim=c4_in_channels, embedding_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embedding_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embedding_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embedding_dim=embedding_dim)
        
        number_of_blocks = 4
        self.linear_fuse_conv = nn.Conv2d(in_channels=embedding_dim * number_of_blocks,
                                          out_channels=embedding_dim,
                                          kernel_size=1)
        self.linear_fuse_norm = nn.SyncBatchNorm(num_features=embedding_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.out = nn.Conv2d(in_channels=embedding_dim, out_channels=num_classes, kernel_size=1)
    
    def apply_linear_fuse(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        output = self.linear_fuse_conv(torch.cat([c4, c3, c2, c1], dim=1))
        output = self.linear_fuse_norm(output)
        output = F.relu(output)
        return output
    
    def forward(self, inputs: list) -> torch.Tensor:
        """
        Args:
            inputs: list of tensors from the output of the MixVisionTransformer
        Returns:
            x: (B, H * W, num_classes)
        """
        c1, c2, c3, c4 = inputs
        
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.apply_linear_fuse(c1=_c1, c2=_c2, c3=_c3, c4=_c4)

        x = self.dropout(_c)
        x = self.out(x)

        return x

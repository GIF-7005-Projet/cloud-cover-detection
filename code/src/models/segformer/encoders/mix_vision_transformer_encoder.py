import torch
import torch.nn as nn
from models.segformer.encoders.overlap_patch_embeddings import OverlapPatchEmbeddings
from models.segformer.encoders.transformer_block import TransformerBlock


class MixVisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channels: int = 4,
        num_classes: int = 1,
        embedding_dims: list = [64, 128, 256, 512],
        num_heads: list = [1, 2, 4, 8],
        mlp_ratios: list = [4, 4, 4, 4],
        qkv_bias: bool = False,
        attention_dropout: float = 0.,
        dropout: float = 0.,
        reduction_ratios: list = [8, 4, 2, 1],
        depths: list = [3, 4, 6, 3]
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        
        self.patch_embed1 = OverlapPatchEmbeddings(image_size=image_size,
                                                   patch_size=7,
                                                   stride=4,
                                                   in_channels=in_channels,
                                                   embedding_dim=embedding_dims[0])
        
        self.patch_embed2 = OverlapPatchEmbeddings(image_size=image_size // 4,
                                                   patch_size=3,
                                                   stride=2,
                                                   in_channels=embedding_dims[0],
                                                   embedding_dim=embedding_dims[1])
        
        self.patch_embed3 = OverlapPatchEmbeddings(image_size=image_size // 8,
                                                   patch_size=3,
                                                   stride=2,
                                                   in_channels=embedding_dims[1],
                                                   embedding_dim=embedding_dims[2])
        
        self.patch_embed4 = OverlapPatchEmbeddings(image_size=image_size // 16,
                                                   patch_size=3,
                                                   stride=2,
                                                   in_channels=embedding_dims[2],
                                                   embedding_dim=embedding_dims[3])
        
        self.block1 = nn.ModuleList([
            TransformerBlock(d_model=embedding_dims[0],
                             num_heads=num_heads[0],
                             mlp_ratio=mlp_ratios[0],
                             qkv_bias=qkv_bias,
                             attention_dropout=attention_dropout,
                             dropout=dropout,
                             reduction_ratio=reduction_ratios[0]) for _ in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(normalized_shape=embedding_dims[0])
        
        self.block2 = nn.ModuleList([
            TransformerBlock(d_model=embedding_dims[1],
                             num_heads=num_heads[1],
                             mlp_ratio=mlp_ratios[1],
                             qkv_bias=qkv_bias,
                             attention_dropout=attention_dropout,
                             dropout=dropout,
                             reduction_ratio=reduction_ratios[1]) for _ in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(normalized_shape=embedding_dims[1])
        
        self.block3 = nn.ModuleList([
            TransformerBlock(d_model=embedding_dims[2],
                             num_heads=num_heads[2],
                             mlp_ratio=mlp_ratios[2],
                             qkv_bias=qkv_bias,
                             attention_dropout=attention_dropout,
                             dropout=dropout,
                             reduction_ratio=reduction_ratios[2]) for _ in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(normalized_shape=embedding_dims[2])
        
        self.block4 = nn.ModuleList([
            TransformerBlock(d_model=embedding_dims[3],
                             num_heads=num_heads[3],
                             mlp_ratio=mlp_ratios[3],
                             qkv_bias=qkv_bias,
                             attention_dropout=attention_dropout,
                             dropout=dropout,
                             reduction_ratio=reduction_ratios[3]) for _ in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(normalized_shape=embedding_dims[3])
    
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        outputs = []
        
        x = self.compute_stage_1(x, B)
        outputs.append(x)
        
        x = self.compute_stage_2(x, B)
        outputs.append(x)
        
        x = self.compute_stage_3(x, B)
        outputs.append(x)
        
        x = self.compute_stage_4(x, B)
        outputs.append(x)
        
        return outputs
    
    def compute_stage_1(self, x: torch.Tensor, B: int) -> torch.Tensor:
        x, H, W = self.patch_embed1(x)
        
        for transformer_block in self.block1:
            x = transformer_block(x, H, W)
        
        x = self.norm1(x)
        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        return x
    
    def compute_stage_2(self, x: torch.Tensor, B: int) -> torch.Tensor:
        x, H, W = self.patch_embed2(x)
        
        for transformer_block in self.block2:
            x = transformer_block(x, H, W)
        
        x = self.norm2(x)
        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        return x
    
    def compute_stage_3(self, x: torch.Tensor, B: int) -> torch.Tensor:
        x, H, W = self.patch_embed3(x)
        
        for transformer_block in self.block3:
            x = transformer_block(x, H, W)
        
        x = self.norm3(x)
        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        return x
    
    def compute_stage_4(self, x: torch.Tensor, B: int) -> torch.Tensor:
        x, H, W = self.patch_embed4(x)
        
        for transformer_block in self.block4:
            x = transformer_block(x, H, W)
        
        x = self.norm4(x)
        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        return x

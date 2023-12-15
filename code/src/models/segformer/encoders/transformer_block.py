import torch
import torch.nn as nn
from models.segformer.encoders.efficient_self_attention import EfficientSelfAttention
from models.segformer.encoders.mix_ffn import MixFFN


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        attention_dropout: float = 0.,
        dropout: float = 0.,
        reduction_ratio: int = 1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        
        self.attention = EfficientSelfAttention(d_model=d_model,
                                                num_heads=num_heads,
                                                qkv_bias=qkv_bias,
                                                attention_dropout=attention_dropout,
                                                projection_dropout=dropout,
                                                reduction_ratio=reduction_ratio)
        
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mix_ffn = MixFFN(in_features=d_model,
                              hidden_features=mlp_hidden_dim,
                              dropout=dropout)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
            H: int
            W: int
        Returns:
            x: (B, N, C)
        """
        x = x + self.attention(self.norm1(x), H, W)
        x = x + self.mix_ffn(self.norm2(x), H, W)
        return x

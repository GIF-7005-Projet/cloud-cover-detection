import torch
import torch.nn as nn


class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attention_dropout: float = 0.,
        projection_dropout: float = 0.,
        reduction_ratio: int = 1
    ):
        super().__init__()
        # Based on original paper (Attention is all you need), d_model must be divisible by num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads}) !"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.scale = (d_model // num_heads) ** -0.5 # scale factor from original paper (Attention is all you need)
        
        self.q = nn.Linear(in_features=d_model,
                           out_features=d_model,
                           bias=qkv_bias)
        self.k = nn.Linear(in_features=d_model,
                           out_features=d_model,
                           bias=qkv_bias)
        self.v = nn.Linear(in_features=d_model,
                           out_features=d_model,
                           bias=qkv_bias)
        
        self.attention_dropout = nn.Dropout(p=attention_dropout)
        
        self.projection = nn.Linear(in_features=d_model,
                                    out_features=d_model)
        self.projection_dropout = nn.Dropout(p=projection_dropout)
        
        self.reduction_ratio = reduction_ratio
        if reduction_ratio > 1:
            self.reduction = nn.Conv2d(in_channels=d_model,
                                       out_channels=d_model,
                                       kernel_size=reduction_ratio,
                                       stride=reduction_ratio)
            self.reduction_layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
            H: int
            W: int
        Returns:
            x: (B, N, C)
        """
        B, N, C = x.shape
        
        queries = self.q(x) # (B, N, C)
        queries = queries.reshape(B, N, self.num_heads, C // self.num_heads) # (B, N, num_heads, C // num_heads)
        queries = queries.permute(0, 2, 1, 3) # (B, num_heads, N, C // num_heads)
        
        if self.reduction_ratio > 1:
            x_reshaped = x.permute(0, 2, 1).reshape(B, C, H, W) # (B, C, H, W)
            x_reduced = self.reduction(x_reshaped).reshape(B, C, -1).permute(0, 2, 1) # (B, N_reduced, C)
            x_reduced = self.reduction_layer_norm(x_reduced) # (B, N_reduced, C)
            
            keys = self.k(x_reduced) # (B, N_reduced, C)
            keys = keys.reshape(B, -1, self.num_heads, C // self.num_heads) # (B, N_reduced, num_heads, C // num_heads)
            keys = keys.permute(0, 2, 1, 3) # (B, num_heads, N_reduced, C // num_heads)
            
            values = self.v(x_reduced) # (B, N_reduced, C)
            values = values.reshape(B, -1, self.num_heads, C // self.num_heads) # (B, N_reduced, num_heads, C // num_heads)
            values = values.permute(0, 2, 1, 3) # (B, num_heads, N_reduced, C // num_heads)
        else:
            keys = self.k(x) # (B, N, C)
            keys = keys.reshape(B, -1, self.num_heads, C // self.num_heads) # (B, N, num_heads, C // num_heads)
            keys = keys.permute(0, 2, 1, 3) # (B, num_heads, N, C // num_heads)
            
            values = self.v(x) # (B, N, C)
            values = values.reshape(B, -1, self.num_heads, C // self.num_heads) # (B, N, num_heads, C // num_heads)
            values = values.permute(0, 2, 1, 3) # (B, num_heads, N, C // num_heads)
        
        keys = keys.transpose(-2, -1) # (B, num_heads, C // num_heads, N)
        
        attention = queries.matmul(keys) * self.scale # (B, num_heads, N, N)
        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)
        
        x = attention.matmul(values).transpose(1, 2).reshape(B, N, C) # (B, N, C)
        x = self.projection(x)
        x = self.projection_dropout(x)
        
        return x

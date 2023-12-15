import torch
import torch.nn as nn
from models.segformer.encoders.mix_vision_transformer_encoder import MixVisionTransformerEncoder
from models.segformer.decoders.semantic_segmentation_head import SegFormerSemanticSegmentationHead
from models.segformer.resize import resize


class SegFormer(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        num_classes: int = 2,
        in_channels: int = 4,
        encoder_embedding_dims: list = [32, 64, 160, 256],
        encoder_reduction_ratios: list = [8, 4, 2, 1],
        encoder_num_heads: list = [1, 2, 5, 8],
        encoder_stages_layers: list = [2, 2, 2, 2],
        decoder_embedding_dim: int = 256,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.encoder_embedding_dims = encoder_embedding_dims
        self.encoder_reduction_ratios = encoder_reduction_ratios
        self.encoder_num_heads = encoder_num_heads
        self.decoder_embedding_dim = decoder_embedding_dim
        
        self.encoder = MixVisionTransformerEncoder(image_size=image_size,
                                                   in_channels=in_channels,
                                                   num_classes=num_classes,
                                                   embedding_dims=encoder_embedding_dims,
                                                   num_heads=encoder_num_heads,
                                                   mlp_ratios=[4, 4, 4, 4],
                                                   qkv_bias=False,
                                                   attention_dropout=0.5,
                                                   dropout=0.5,
                                                   reduction_ratios=encoder_reduction_ratios,
                                                   encoder_stages_layers=encoder_stages_layers)
        
        self.decoder = SegFormerSemanticSegmentationHead(c1_in_channels=encoder_embedding_dims[0],
                                                         c2_in_channels=encoder_embedding_dims[1],
                                                         c3_in_channels=encoder_embedding_dims[2],
                                                         c4_in_channels=encoder_embedding_dims[3],
                                                         num_classes=num_classes,
                                                         embedding_dim=decoder_embedding_dim,
                                                         dropout=0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return resize(input=x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

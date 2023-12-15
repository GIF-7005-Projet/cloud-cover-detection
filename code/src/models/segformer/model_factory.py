from models.segformer.model import SegFormer


def create_b0_model(image_size: int, num_classes: int, in_channels: int) -> SegFormer:
    return SegFormer(
        image_size = image_size,
        num_classes = num_classes,
        in_channels = in_channels,
        encoder_embedding_dims = [32, 64, 160, 256],
        encoder_reduction_ratios = [8, 4, 2, 1],
        encoder_num_heads = [1, 2, 5, 8],
        encoder_stages_layers = [2, 2, 2, 2],
        decoder_embedding_dim = 256
    )

from models.segformer.lightning_module import LightningSegFormer


def create_b0_model(
    image_size: int, 
    num_classes: int, 
    in_channels: int, 
    learning_rate: float,
    max_epochs: int
) -> LightningSegFormer:
    return LightningSegFormer(
        image_size = image_size,
        n_classes = num_classes,
        in_channels = in_channels,
        encoder_embedding_dims = [32, 64, 160, 256],
        encoder_reduction_ratios = [8, 4, 2, 1],
        encoder_num_heads = [1, 2, 5, 8],
        encoder_stages_layers = [2, 2, 2, 2],
        encoder_qkv_bias = True,
        encoder_dropout = 0.,
        decoder_embedding_dim = 256,
        decoder_dropout = 0.,
        learning_rate = learning_rate,
        max_epochs = max_epochs
    )


def create_b1_model(
    image_size: int, 
    num_classes: int, 
    in_channels: int, 
    learning_rate: float,
    max_epochs: int
) -> LightningSegFormer:
    return LightningSegFormer(
        image_size = image_size,
        n_classes = num_classes,
        in_channels = in_channels,
        encoder_embedding_dims = [64, 128, 320, 512],
        encoder_reduction_ratios = [8, 4, 2, 1],
        encoder_num_heads = [1, 2, 5, 8],
        encoder_stages_layers = [2, 2, 2, 2],
        encoder_qkv_bias = True,
        encoder_dropout = 0.,
        decoder_embedding_dim = 256,
        decoder_dropout = 0.,
        learning_rate = learning_rate,
        max_epochs = max_epochs
    )


def create_b2_model(
    image_size: int, 
    num_classes: int, 
    in_channels: int, 
    learning_rate: float,
    max_epochs: int
) -> LightningSegFormer:
    return LightningSegFormer(
        image_size = image_size,
        n_classes = num_classes,
        in_channels = in_channels,
        encoder_embedding_dims = [64, 128, 320, 512],
        encoder_reduction_ratios = [8, 4, 2, 1],
        encoder_num_heads = [1, 2, 5, 8],
        encoder_stages_layers = [3, 3, 6, 3],
        encoder_qkv_bias = True,
        encoder_dropout = 0.,
        decoder_embedding_dim = 768,
        decoder_dropout = 0.,
        learning_rate = learning_rate,
        max_epochs = max_epochs
    )


def create_b3_model(
    image_size: int, 
    num_classes: int, 
    in_channels: int, 
    learning_rate: float,
    max_epochs: int
) -> LightningSegFormer:
    return LightningSegFormer(
        image_size = image_size,
        n_classes = num_classes,
        in_channels = in_channels,
        encoder_embedding_dims = [64, 128, 320, 512],
        encoder_reduction_ratios = [8, 4, 2, 1],
        encoder_num_heads = [1, 2, 5, 8],
        encoder_stages_layers = [3, 3, 18, 3],
        encoder_qkv_bias = True,
        encoder_dropout = 0.,
        decoder_embedding_dim = 768,
        decoder_dropout = 0.,
        learning_rate = learning_rate,
        max_epochs = max_epochs
    )


def create_b4_model(
    image_size: int, 
    num_classes: int, 
    in_channels: int, 
    learning_rate: float,
    max_epochs: int
) -> LightningSegFormer:
    return LightningSegFormer(
        image_size = image_size,
        n_classes = num_classes,
        in_channels = in_channels,
        encoder_embedding_dims = [64, 128, 320, 512],
        encoder_reduction_ratios = [8, 4, 2, 1],
        encoder_num_heads = [1, 2, 5, 8],
        encoder_stages_layers = [3, 8, 27, 3],
        encoder_qkv_bias = True,
        encoder_dropout = 0.,
        decoder_embedding_dim = 768,
        decoder_dropout = 0.,
        learning_rate = learning_rate,
        max_epochs = max_epochs
    )


def create_b5_model(
    image_size: int, 
    num_classes: int, 
    in_channels: int, 
    learning_rate: float,
    max_epochs: int
) -> LightningSegFormer:
    return LightningSegFormer(
        image_size = image_size,
        n_classes = num_classes,
        in_channels = in_channels,
        encoder_embedding_dims = [64, 128, 320, 512],
        encoder_reduction_ratios = [8, 4, 2, 1],
        encoder_num_heads = [1, 2, 5, 8],
        encoder_stages_layers = [3, 6, 40, 3],
        encoder_qkv_bias = True,
        encoder_dropout = 0.,
        decoder_embedding_dim = 768,
        decoder_dropout = 0.,
        learning_rate = learning_rate,
        max_epochs = max_epochs
    )

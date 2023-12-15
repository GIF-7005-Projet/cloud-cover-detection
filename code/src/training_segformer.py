from datasets.data_module import CloudCoverDataModule
from pathlib import Path
from models.segformer.lightning_module import LightningSegFormer
from training.trainer import train
from testing.tester import test


if __name__ == '__main__':
    data_module = CloudCoverDataModule(
        train_X_folder_path=Path("../../data/final/public/train_features/"),
        train_y_folder_path=Path("../../data/final/public/train_labels/"),
        test_X_folder_path=Path("../../data/final/private/test_features/"),
        test_y_folder_path=Path("../../data/final/private/test_labels/"),
        train_batch_size=8,
        val_batch_size=16,
        test_batch_size=16,
        val_size=0.2,
        random_state=42
    )

    data_module.prepare_data()

    data_module.setup(stage="fit")
    data_module.setup(stage="test")

    deeplab = train(
        model=LightningSegFormer(
            image_size=512,
            n_classes=2,
            in_channels=4,
            encoder_embedding_dims=[32, 64, 160, 256],
            encoder_reduction_ratios=[8, 4, 2, 1],
            encoder_num_heads=[1, 2, 5, 8],
            encoder_stages_layers=[2, 2, 2, 2],
            decoder_embedding_dim=256,
            learning_rate=1e-3
        ),
        run_name="segformer_b0",
        model_version=0,
        data_module=data_module,
        max_epochs=2,
        patience=10
    )

    test(
        model=deeplab,
        run_name="segformer_b0",
        model_version=0,
        data_module=data_module
    )

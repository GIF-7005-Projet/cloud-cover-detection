from datasets.data_module import CloudCoverDataModule
from pathlib import Path
from models.unet import LightningUNet
from training.trainer import train
from testing.tester import test


if __name__ == '__main__':
    data_module = CloudCoverDataModule(
        train_X_folder_path=Path("../../data/final/public/train_features/"),
        train_y_folder_path=Path("../../data/final/public/train_labels/"),
        test_X_folder_path=Path("../../data/final/private/test_features/"),
        test_y_folder_path=Path("../../data/final/private/test_labels/"),
        train_batch_size=4,
        val_batch_size=8,
        test_batch_size=8,
        val_size=0.2,
        random_state=42
    )

    data_module.prepare_data()

    data_module.setup(stage="fit")
    data_module.setup(stage="test")

    unet = train(
        model=LightningUNet(n_channels=4, n_classes=2, bilinear=True, learning_rate=0.001),
        run_name="unet",
        model_version=0,
        data_module=data_module,
        max_epochs=35,
        patience=5
    )

    #unet = LightningUNet.load_from_checkpoint(checkpoint_path='./logs/unet/version_0/checkpoints/unet-epoch=07-val_loss=0.26.ckpt', n_channels=4, n_classes=2)

    test(
        model=unet,
        run_name="unet",
        model_version=0,
        data_module=data_module
    )

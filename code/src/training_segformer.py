from datasets.data_module import CloudCoverDataModule
from pathlib import Path
from models.segformer.model_factory import (
    create_b0_model,
    create_b1_model,
    create_b2_model,
    create_b3_model,
    create_b4_model,
    create_b5_model
)
from training.trainer import train
from testing.tester import test

MAX_EPOCHS = 20
PATIENCE = 10
LR = 6e-5

if __name__ == '__main__':
    data_module = CloudCoverDataModule(
        train_X_folder_path=Path("../../data/final/public/train_features/"),
        train_y_folder_path=Path("../../data/final/public/train_labels/"),
        test_X_folder_path=Path("../../data/final/private/test_features/"),
        test_y_folder_path=Path("../../data/final/private/test_labels/"),
        train_batch_size=16,
        val_batch_size=32,
        test_batch_size=32,
        val_size=0.2,
        random_state=42
    )

    data_module.prepare_data()

    data_module.setup(stage="fit")
    data_module.setup(stage="test")
    
    segformer = create_b0_model(
        image_size=512,
        num_classes=2,
        in_channels=4,
        learning_rate=LR,
        max_epochs=MAX_EPOCHS
    )

    segformer = train(
        model=segformer,
        run_name="segformer_b0",
        model_version=0,
        data_module=data_module,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        monitor_lr=True
    )

    test(
        model=segformer,
        run_name="segformer_b0",
        model_version=0,
        data_module=data_module
    )

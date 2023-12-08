import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


def test(
        model: pl.LightningModule,
        run_name: str,
        model_version: int,
        data_module: pl.LightningDataModule
):
    logger = TensorBoardLogger(
        save_dir='logs',
        name=run_name,
        version=model_version
    )

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=logger
    )

    trainer.test(model, dataloaders=data_module.test_dataloader())

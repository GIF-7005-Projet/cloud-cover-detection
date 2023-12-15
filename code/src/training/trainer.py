import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

def train(
        model: pl.LightningModule,
        run_name: str,
        model_version: int,
        data_module: pl.LightningDataModule,
        max_epochs: int,
        patience: int,
        monitor_lr: bool = False
    ) -> pl.LightningModule:

    logger = TensorBoardLogger(
        save_dir='logs',
        name=run_name,
        version=model_version
    )
    
    early_stopping = EarlyStopping('val_loss', patience=patience)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename=run_name+'-{epoch:02d}-{val_loss:.2f}')
    
    callbacks = [
        early_stopping,
        checkpoint_callback
    ]
    
    if monitor_lr:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        log_every_n_steps=10
    )

    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())

    return model

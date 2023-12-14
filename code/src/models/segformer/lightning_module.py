import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics
from models.segformer.model import SegFormer


class LightningSegFormer(pl.LightningModule):
    def __init__(
        self, 
        image_size: int = 512,
        n_classes: int = 1,
        in_channels: int = 4,
        encoder_embedding_dims: list = [64, 128, 256, 512],
        decoder_embedding_dim: int = 768,
        learning_rate=6e-5
    ):
        super().__init__()
        self.model = SegFormer(
            image_size=image_size,
            num_classes=n_classes,
            in_channels=in_channels,
            encoder_embedding_dims=encoder_embedding_dims,
            decoder_embedding_dim=decoder_embedding_dim
        )
        self.image_size = image_size
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.encoder_embedding_dims = encoder_embedding_dims
        self.decoder_embedding_dim = decoder_embedding_dim
        self.learning_rate = learning_rate

        self.save_hyperparameters()

        self.train_jaccard = torchmetrics.JaccardIndex(num_classes=n_classes, task='binary')
        self.train_accuracy = torchmetrics.Accuracy(num_classes=n_classes, task='binary', average='macro')

        self.val_jaccard = torchmetrics.JaccardIndex(num_classes=n_classes, task='binary')
        self.val_accuracy = torchmetrics.Accuracy(num_classes=n_classes, task='binary', average='macro')

        self.test_jaccard = torchmetrics.JaccardIndex(num_classes=n_classes, task='binary')
        self.test_accuracy = torchmetrics.Accuracy(num_classes=n_classes, task='binary', average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        y_hat = self(inputs)
        predicted_labels = torch.argmax(y_hat, dim=1)

        loss = self.compute_loss(y_hat, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.train_jaccard(predicted_labels, target)
        self.log('train_jaccard', self.train_jaccard, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.train_accuracy(predicted_labels, target)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def compute_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, target = batch
        y_hat = self(inputs)
        predicted_labels = torch.argmax(y_hat, dim=1)

        loss = self.compute_loss(y_hat, target)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.val_jaccard(predicted_labels, target)
        self.log('val_jaccard', self.val_jaccard, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.val_accuracy(predicted_labels, target)
        self.log('val_accuracy', self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        y_hat = self(inputs)
        predicted_labels = torch.argmax(y_hat, dim=1)

        loss = self.compute_loss(y_hat, target)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.test_jaccard(predicted_labels, target)
        self.log('test_jaccard', self.test_jaccard, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.test_accuracy(predicted_labels, target)
        self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
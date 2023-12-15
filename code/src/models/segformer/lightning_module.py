import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics
from torch.optim.lr_scheduler import PolynomialLR
from models.segformer.model import SegFormer


class LightningSegFormer(pl.LightningModule):
    def __init__(
        self, 
        image_size: int = 512,
        n_classes: int = 1,
        in_channels: int = 4,
        encoder_embedding_dims: list = [32, 64, 160, 256],
        encoder_reduction_ratios: list = [8, 4, 2, 1],
        encoder_num_heads: list = [1, 2, 5, 8],
        encoder_stages_layers: list = [2, 2, 2, 2],
        encoder_qkv_bias: bool = True,
        encoder_dropout: float = 0.,
        decoder_embedding_dim: int = 256,
        decoder_dropout: float = 0.,
        learning_rate = 6e-5,
        max_epochs: int = 20
    ):
        super().__init__()
        self.model = SegFormer(
            image_size=image_size,
            num_classes=n_classes,
            in_channels=in_channels,
            encoder_embedding_dims=encoder_embedding_dims,
            encoder_reduction_ratios=encoder_reduction_ratios,
            encoder_num_heads=encoder_num_heads,
            encoder_stages_layers=encoder_stages_layers,
            decoder_embedding_dim=decoder_embedding_dim
        )
        self.image_size = image_size
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.encoder_embedding_dims = encoder_embedding_dims
        self.encoder_reduction_ratios = encoder_reduction_ratios
        self.encoder_num_heads = encoder_num_heads
        self.encoder_stages_layers = encoder_stages_layers
        self.encoder_qkv_bias = encoder_qkv_bias
        self.encoder_dropout = encoder_dropout
        self.decoder_embedding_dim = decoder_embedding_dim
        self.decoder_dropout = decoder_dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

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
        self.log("hp_metric", loss)
        
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
        scheduler = PolynomialLR(optimizer, total_iters=self.max_epochs, power=1.0, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

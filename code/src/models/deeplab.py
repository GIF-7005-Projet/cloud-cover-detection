import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics


# MODELE DEEPLAB
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # Dilations par incrément de 6. On peut ajouter 24 dans la liste au besoin
        self.dilations = [1, 6, 12, 18]
        self.aspp_blocks = nn.ModuleList()

        for dilation in self.dilations:
            self.aspp_blocks.append(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
            )
        self.output_conv = nn.Conv2d(len(self.dilations) * out_channels, out_channels, 1)

    def forward(self, x):
        aspp_outputs = [block(x) for block in self.aspp_blocks]

        x = torch.cat(aspp_outputs, dim=1)

        x = self.output_conv(x)

        return x

# Cette fonction vise à adapter le modèle Resnet pour 4 channels
def modify_resnet4channel(in_channels=4):
    model = models.resnet50(pretrained=True)

    # On modifie la première convolution
    old_conv_layer = model.conv1
    new_conv_layer = nn.Conv2d(in_channels, old_conv_layer.out_channels,
                               kernel_size=old_conv_layer.kernel_size,
                               stride=old_conv_layer.stride,
                               padding=old_conv_layer.padding,
                               bias=False)

    # On transfère les poids vers le nouveau modèle
    with torch.no_grad():
        new_conv_layer.weight[:, :3, :, :] = old_conv_layer.weight.clone()
        mean_weights = torch.mean(old_conv_layer.weight, dim=1)
        repeated_mean_weights = mean_weights.repeat(1, 1, 1, 1)
        new_conv_layer.weight[:, 3, :, :] = repeated_mean_weights

    model.conv1 = new_conv_layer

    # Cette ligne enlève les deux dernières étapes (av. pooling et fully connected layers)
    model = nn.Sequential(*list(model.children())[:-2])

    return model

class DeepLabV3(nn.Module):
    def __init__(self, num_classes, in_channels=4):
        super(DeepLabV3, self).__init__()

        # On a besoin d'un modèle backbone. J'ai utilisé Resnet
        self.backbone = modify_resnet4channel(in_channels)

        # Atrous
        self.aspp = ASPP(2048, 256)

        self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x = self.backbone(x)

        x = self.aspp(x)

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)

        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)
        return x


class LightningDeeplab(pl.LightningModule):
    def __init__(self, n_channels, n_classes, bilinear=True, learning_rate=1e-3):
        super().__init__()
        self.model = DeepLabV3(n_classes, n_channels)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
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
        y_hat = F.interpolate(y_hat, size=target.size()[1:], mode='bilinear', align_corners=False)
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
        y_hat = F.interpolate(y_hat, size=target.size()[1:], mode='bilinear', align_corners=False)
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
        y_hat = F.interpolate(y_hat, size=target.size()[1:], mode='bilinear', align_corners=False)
        predicted_labels = torch.argmax(y_hat, dim=1)

        loss = self.compute_loss(y_hat, target)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.test_jaccard(predicted_labels, target)
        self.log('test_jaccard', self.test_jaccard, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.test_accuracy(predicted_labels, target)
        self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
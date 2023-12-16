import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics
from torchvision.models.detection import maskrcnn_resnet50_fpn, backbone_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(n_classes, n_channels) :
    model = maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,n_classes)
    model.backbone.body.conv1 = nn.Conv2d(in_channels=n_channels,
                            out_channels=model.backbone.body.conv1.out_channels,
                            kernel_size = model.backbone.body.conv1.kernel_size,
                            stride=model.backbone.body.conv1.stride,
                            padding=model.backbone.body.conv1.padding,
                            dilation=model.backbone.body.conv1.dilation,
                            groups=model.backbone.body.conv1.groups,
                            bias= (model.backbone.body.conv1.bias is not None))
    
    with torch.no_grad():
        model.backbone.body.conv1.weight[:, :4, :, :] = model.backbone.body.conv1.weight[:, :4, :, :]
        # if  model.backbone.body.conv1.bias is not None :
        #     model.conv1.bias[:] =  model.conv1.bias[:]

    return model 

class MaskRCnn(pl.LightningModule) :
    def __init__(self, n_channels, n_classes, bilinear=True, learning_rate=1e-3):
        super().__init__()
        self.model = create_model(n_classes, n_channels) 
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
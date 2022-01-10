import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torchvision.models.video import r3d_18


class ViolenceDetection(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = r3d_18(pretrained=True)
        self.model.fc = nn.Linear(512, 1)
        self.criterion = nn.BCELoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
    
    def forward(self, x):
        out = self.model(torch.permute(x, (0,2,1,3,4)))
        out = torch.sigmoid(out)
        return out
    
    def training_step(self, batch, batch_idx):
        # B x T x C x H x W
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.view(-1)
        loss = self.criterion(y_hat, y.float())
        threshold = torch.tensor([0.5]).type_as(y_hat)
        y_hat = (y_hat>threshold).float()*1
        self.train_accuracy(y_hat, y)

        return loss
    
    def training_epoch_end(self, outputs):
        train_loss = torch.tensor([dict["loss"] for dict in outputs]).mean()
        epoch_acc = self.train_accuracy.compute()
        self.train_accuracy.reset()
        self.log('train_loss', train_loss)
        self.log('train_acc', epoch_acc)
        with open('./train_log.csv', 'a') as f:
            f.write(f'Train Epoch: {self.current_epoch}, Accuracy: {epoch_acc}, Loss: {train_loss}\n')
    
    def validation_step(self, batch, batch_idx):
        # B x T x C x H x W
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.view(-1)
        loss = self.criterion(y_hat, y.float())
        threshold = torch.tensor([0.5]).type_as(y_hat)
        y_hat = (y_hat>threshold).float()*1
        self.valid_accuracy(y_hat, y)

        return loss
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.tensor(outputs).mean()
        epoch_acc = self.valid_accuracy.compute()
        self.valid_accuracy.reset()
        self.log('val_loss', val_loss)
        self.log('val_acc', epoch_acc)
        with open('./train_log.csv', 'a') as f:
            f.write(f'Validation Epoch: {self.current_epoch}, Accuracy: {epoch_acc}, Loss: {val_loss}\n')
    
    def test_step(self, batch, batch_idx):
        # B x T x C x H x W
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.view(-1)
        loss = self.criterion(y_hat, y.float())
        threshold = torch.tensor([0.5]).type_as(y_hat)
        y_hat = (y_hat>threshold).float()*1
        self.valid_accuracy(y_hat, y)
 
        return loss
    
    def test_epoch_end(self, outputs):
        val_loss = torch.tensor(outputs).mean()
        epoch_acc = self.valid_accuracy.compute()
        self.valid_accuracy.reset()
        self.log('test_loss', val_loss)
        self.log('test_acc', epoch_acc)
        with open('./train_log.csv', 'a') as f:
            f.write(f'Test Epoch: {self.current_epoch}, Accuracy: {epoch_acc}, Loss: {val_loss}\n')
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')
        with open('./train_log.csv', 'a') as f:
            f.write(f'Parameters Learning rate: {0.0001}, Momentum: {0.9}, Weight decay: {0.0001}\n')
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }
    
    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_acc", min_delta=0.001, patience=5, verbose=False, mode="max")

        checkpoint = ModelCheckpoint(
                dirpath='./',
                filename='{epoch}-{train_loss:.3f}-{train_acc:.3f}-{val_loss:.3f}-{val_acc:.3f}',
                monitor="val_acc",
                mode='max')

        lr_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum=True)
        return [early_stop, lr_monitor, checkpoint]
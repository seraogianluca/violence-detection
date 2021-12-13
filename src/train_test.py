import torch
from torch.nn.functional import threshold
from torch.optim import lr_scheduler
import torchvision
import torchmetrics
import pytorch_lightning as pl
import torch.nn as nn

from typing import Optional

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data.dataloader import DataLoader
from data.video import SpatioTemporalDataset
from data.utils import stratified_random_split
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
        x = torch.permute(x, (0,2,1,3,4))
        out = self.model(x)
        return out
    
    def training_step(self, batch, batch_idx):
        # B x T x C x H x W
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(0), y.float())
        threshold = torch.tensor([0.5]).type_as(y_hat)
        y_hat = (y_hat>threshold).float()*1
        self.train_accuracy(y_hat.squeeze(0), y)
        return loss
    
    def training_epoch_end(self, outputs):
        train_loss = torch.tensor([dict["loss"] for dict in outputs]).mean()
        epoch_acc = self.train_accuracy.compute()
        self.train_accuracy.reset()
        self.log('train_loss', train_loss, prog_bar=True)
        self.log('train_acc', epoch_acc, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        # B x T x C x H x W
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(0), y.float())
        threshold = torch.tensor([0.5]).type_as(y_hat)
        y_hat = (y_hat>threshold).float()*1
        self.valid_accuracy(y_hat.squeeze(0), y)
        return loss
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.tensor(outputs).mean()
        epoch_acc = self.valid_accuracy.compute()
        self.valid_accuracy.reset()
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', epoch_acc, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            },
        }

class ViolenceDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'path/to/dir', batch_size: int = 1, num_clips: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_clips = num_clips
    
    def setup(self, stage: Optional[str] = None):
        violence_full = SpatioTemporalDataset(self.data_dir, num_clips=self.num_clips)
        self.train, self.val = stratified_random_split(violence_full, (0.7,0.3), violence_full.targets())

        train_pipeline = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(112),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989))])
        self.train.dataset.transforms = train_pipeline
        self.train.dataset.train = True

        valid_pipeline = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(112),
            torchvision.transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989))])
        self.val.dataset.transforms = valid_pipeline
        self.val.dataset.train = False
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=8)

if __name__ == '__main__':
    data = ViolenceDataset('/mnt/d/serao/fight_detection', num_clips=8, batch_size=1)

    model = ViolenceDetection()

    es = EarlyStopping(monitor="val_acc", min_delta=0.001, patience=3, verbose=False, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum=True)

    trainer = pl.Trainer(callbacks=[es, lr_monitor], gpus=1, max_epochs=60, num_sanity_val_steps=0)
    trainer.fit(model, data)
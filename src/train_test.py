import torch
import torchvision
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data.dataloader import DataLoader
from data.video import SpatioTemporalDataset
from data.utils import stratified_random_split
from torchvision.models.video import r3d_18

class ViolenceDetection(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = r3d_18(pretrained=True)
        self.model.fc = nn.Linear(512, 1, bias=True)
    
    def forward(self, x):
        return torch.sigmoid(self.model(torch.permute(x, (0, 2, 1, 3, 4))))
    
    def training_step(self, batch, batch_idx):
        # B x T x C x H x W
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.float().unsqueeze(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        # B x T x C x H x W
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.float().unsqueeze(0))
        return loss
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)

class ViolenceDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'path/to/dir', batch_size: int = 1, num_clips: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
    
    def setup(self, stage: Optional[str] = None):
        pipeline = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(112),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989))])

        violence_full = SpatioTemporalDataset(
            '/mnt/d/serao/archive/Real Life Violence Dataset',
            transforms=pipeline,
            num_clips=16)
        self.train, self.val = stratified_random_split(violence_full, (0.7,0.3), violence_full.targets)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=8)

if __name__ == '__main__':
    data = ViolenceDataset('/workdir/data/archive/Real Life Violence Dataset', batch_size=1)
    #model = ViolenceDetection()

    #early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")

    #trainer = pl.Trainer(callbacks=[early_stop_callback], gpus=1)
    #trainer.fit(model, data)
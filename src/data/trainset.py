
import torchvision
import pytorch_lightning as pl

from torch.utils.data.dataloader import DataLoader
from data.video import SpatioTemporalDataset, OpticalFlowDataset
from data.utils import stratified_random_split


class ViolenceDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'path/to/dir', batch_size: int = 1, num_clips: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_clips = num_clips
    
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            violence_full = SpatioTemporalDataset(self.data_dir, num_clips=self.num_clips)
            self.train, self.val = stratified_random_split(violence_full, (0.8,0.2), violence_full.targets())

            train_pipeline = torchvision.transforms.Compose([
                torchvision.transforms.Resize(112),
                torchvision.transforms.CenterCrop(112),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989))])
            self.train.dataset.transforms = train_pipeline
            self.train.dataset.train = True

            valid_pipeline = torchvision.transforms.Compose([
                torchvision.transforms.Resize(112),
                torchvision.transforms.CenterCrop(112),
                torchvision.transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989))])
            self.val.dataset.transforms = valid_pipeline
            self.val.dataset.train = False
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=8)

class ViolenceDatasetOF(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'path/to/dir', batch_size: int = 1, num_clips: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_clips = num_clips
    
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            violence_full = OpticalFlowDataset(self.data_dir, num_clips=self.num_clips, size=112)
            self.train, self.val = stratified_random_split(violence_full, (0.8,0.2), violence_full.targets())

            train_pipeline = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip()])
            self.train.dataset.transforms = train_pipeline
            self.train.dataset.train = True

            self.val.dataset.transforms = None
            self.val.dataset.train = False
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=8)
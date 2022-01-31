
import torchvision
import pytorch_lightning as pl

from torch.utils.data.dataloader import DataLoader
from data.video import SpatioTemporalDataset
from data.utils import stratified_random_split


class ViolenceTest(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'path/to/dir', batch_size: int = 1, num_clips: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_clips = num_clips
    
    def setup(self, stage):
        pipeline = torchvision.transforms.Compose([
            torchvision.transforms.Resize(112),
            torchvision.transforms.CenterCrop(112),
            torchvision.transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989))])
        self.test = SpatioTemporalDataset(self.data_dir, num_clips=self.num_clips, transforms=pipeline, train=False)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=8)
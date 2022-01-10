import os
import torch
import numpy as np

from torch import Tensor
from typing import Callable, Optional, Tuple, List
from torchvision.io import read_video
from torchvision.datasets.folder import make_dataset
from torchvision.datasets import VisionDataset

import decord
from decord import VideoReader
from decord import cpu

# set decord bridge to have torch output tensors
decord.bridge.set_bridge('torch')

class SpatioTemporalDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.avi', '.mp4'),
        num_clips: int = 16,
        train: bool = True) -> None:

        super().__init__(root)
        self.root = root
        self.transforms = transforms
        self.extensions = extensions

        # find classes from dataset folder
        # make dataset list (path_to_video, class)
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.classes.sort()
        class_to_index = {cls_name: i for i, cls_name in enumerate(self.classes)}
        # list(tuple(str, int))
        self.videos_index = make_dataset(root, class_to_index, extensions, is_valid_file=None)
        self.num_clips = num_clips
        self.train = train
    
    def targets(self):
        targets = [video[1] for video in self.videos_index]
        return targets

    def set_transforms(self, transforms):
        self.transforms = transforms
    
    def set_train(self, train):
        self.train = train

    def _sample_frames(
        self, 
        num_frames: int) -> List[int]:
        # divide a video into num_clip buckets, uniform sampling from each bucket.
        # segment based sampling
        # video tensor is TxCxHxW

        if self.num_clips > num_frames:
            raise ValueError('num_clips should be lower than frames in video')
        
        average_duration = num_frames // self.num_clips
        offsets = np.multiply(list(range(self.num_clips)), average_duration) + np.random.randint(average_duration, size=self.num_clips)

        return offsets
    
    def _validation_frames(
        self, 
        num_frames: int) -> List[int]:

        if self.num_clips > num_frames:
            raise ValueError('num_clips should be lower than frames in video')
        
        average_duration = num_frames // self.num_clips
        offsets = np.multiply(list(range(self.num_clips)), average_duration)

        return offsets

    def __getitem__(
        self, 
        index: int) -> Tuple[Tensor, int]:
        path_to_video, label = self.videos_index[index]

        reader = VideoReader(path_to_video, ctx=cpu(0))

        offsets = []
        if self.train:
            offsets = self._sample_frames(len(reader))
        else:
            offsets = self._validation_frames(len(reader))

        if len(offsets) == 0:
            raise ValueError('could not read the video')

        frames = reader.get_batch(offsets)
        frames = torch.permute(frames, (0, 3, 1, 2))
        frames = frames.type(torch.float32).div(255)

        if self.transforms is not None:
            # consider the entire video as batch of images
            frames = self.transforms(frames)

        return frames, label
    
    def __len__(self):
        return len(self.videos_index)

if __name__ == '__main__':
    pass
    # import torchvision.transforms as T
    # from torch.utils.data.dataloader import DataLoader
    # from tqdm import tqdm
    # import utils as U
 
    # pipeline = T.Compose([
    #     T.RandomResizedCrop((112, 112)),
    #    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
 
    # dataset = SpatioTemporalDataset(
    #      '/mnt/d/serao/real_life_violence',
    #      transforms=pipeline,
    #      num_clips=16
    # )
     
    # itr_dataset = iter(DataLoader(dataset, shuffle=False))
    # for video, label in tqdm(itr_dataset):
    #     pass
 
    # print(f'Dataset size: {len(dataset)}')
 
    # train_set, val_set = U.stratified_random_split(dataset, (0.7, 0.3), dataset.targets)
 
    # print(f'Train size: {len(train_set)}')
    # print(f'Train type: {train_set}')
    # print(f'Validation size: {len(val_set)}')
    # print(f'Validation type: {val_set}')

    # dataloader returns a batch
    # we need to extract the videoclip
    
    # print(f'Video shape: {video.shape}')
    # print(f'Video dtype: {video.dtype}')
    # print(f'Video class: {cls}')
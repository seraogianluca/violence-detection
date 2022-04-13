import os
import torch
import numpy as np
import decord

from torch import Tensor
from typing import Callable, Optional, Tuple, List
from torchvision.datasets.folder import make_dataset
from torchvision.datasets import VisionDataset
from decord import VideoReader
from decord import cpu
from transforms.optflow import opticalFlowDenseRLOF

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
    
    def _sample_frames_uniform(
        self, 
        num_frames: int) -> List[int]:

        if self.num_clips > num_frames:
            raise ValueError('num_clips should be lower than frames in video')

        # uniform sampling without duplicates
        offsets = []
        while len(offsets) != self.num_clips:
            offset = np.random.randint(num_frames, size=1)
            if offset[0] not in offsets:
                offsets.append(offset[0])
        
        return np.array(sorted(offsets))

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
            # offsets = self._sample_frames_uniform(len(reader))
            offsets = self._sample_frames(len(reader))
        else:
            offsets = self._validation_frames(len(reader))

        if len(offsets) == 0:
            raise ValueError('could not read the video')

        frames = reader.get_batch(offsets)
        # frames is TxHxWxC so transformed to TxCxHxW
        # standard format taken by transformation (batch of images)
        frames = torch.permute(frames, (0, 3, 1, 2))
        frames = frames.type(torch.float32).div(255)

        if self.transforms is not None:
            # consider the entire video as batch of images
            frames = self.transforms(frames)

        return path_to_video, frames, label
    
    def __len__(self):
        return len(self.videos_index)

class OpticalFlowDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.avi', '.mp4'),
        num_clips: int = 16,
        train: bool = True,
        size: int = 112) -> None:

        super().__init__(root)
        self.root = root
        self.transforms = transforms
        self.extensions = extensions        
        self.num_clips = num_clips
        self.train = train
        self.size = size

        # find classes from dataset folder
        # make dataset list (path_to_video, class)
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.classes.sort()
        class_to_index = {cls_name: i for i, cls_name in enumerate(self.classes)}
        # list(tuple(str, int))
        self.videos_index = make_dataset(root, class_to_index, extensions, is_valid_file=None)

    def targets(self):
        targets = [video[1] for video in self.videos_index]
        return targets

    def _sample_frames(
        self, 
        num_frames: int) -> List[int]:
        
        average_duration = num_frames // self.num_clips
        offsets = np.multiply(list(range(self.num_clips)), average_duration) + np.random.randint(average_duration, size=self.num_clips)
        ext_offsets = np.add(offsets, 1)
        result = np.sort(np.concatenate((offsets, ext_offsets)))

        return result
    
    def _validation_frames(
        self, 
        num_frames: int) -> List[int]:

        average_duration = num_frames // self.num_clips
        offsets = np.multiply(list(range(self.num_clips)), average_duration)
        ext_offsets = np.add(offsets, 1)
        result = np.sort(np.concatenate((offsets, ext_offsets)))

        return result

    def __getitem__(
        self, 
        index: int) -> Tuple[Tensor, int]:
        path_to_video, label = self.videos_index[index]

        reader = VideoReader(path_to_video, ctx=cpu(0))
        video_len = len(reader)

        if self.num_clips > video_len:
            raise ValueError('num_clips should be lower than frames in video')
        elif (self.num_clips * 2) > video_len:
            raise ValueError('video too short to be transformed into optical flow, it should be at least twice the number of clips')

        offsets = []
        if self.train:
            offsets = self._sample_frames(video_len)
        else:
            offsets = self._validation_frames(video_len)

        frames = reader.get_batch(offsets)
        # frames is TxHxWxC so transformed to TxCxHxW
        # standard format taken by transformation (batch of images)
        frames = torch.permute(frames, (0, 3, 1, 2))
        frames = opticalFlowDenseRLOF(frames, self.size)
        frames = frames.type(torch.float32).div(255)

        if self.transforms is not None:
            # consider the entire video as batch of images
            frames = self.transforms(frames)

        return frames, label
    
    def __len__(self):
        return len(self.videos_index)
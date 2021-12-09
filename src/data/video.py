import os
from numpy.random.mtrand import sample
import torch
import numpy as np

from torch import Tensor
from typing import Callable, Optional, Tuple
from torch._C import dtype
from torchvision.io import read_video
from torchvision.datasets.folder import make_dataset
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader

class VideoRecord:
    def __init__(
        self, 
        path: str, 
        num_frames: int, 
        label: int
        ) -> None:

        self.path = path
        self.num_frames = int(num_frames)
        self.label = int(label)

    @property
    def path(self):
        return self.path
    
    @property
    def num_frames(self):
        return self.num_frames
    
    @property
    def label(self):
        return self.label

# TODO: RGB and OPTICAL FLOW
class VideoDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.avi', '.mp4')
        ) -> None:

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
    
    def targets(self):
        targets = [video[1] for video in self.videos_index]
        return targets

    def __len__(self):
        return len(self.videos_index)
    
    def __getitem__(
        self, 
        index: int) -> Tuple[Tensor, int]:
        # TODO: implements caching for video reading?

        path_to_video, video_cls = self.videos_index[index]

        # video, audio, metadata
        # video frames -> Tensor[T, H, W, C]
        # T number of frames, C channels
        frames, _, _ = read_video(path_to_video)

        # transform frames into batch of images format
        # useful to reuse transformations on images
        # batch of images -> Tensor[B, C, H, W]
        # B batch size, C channels
        frames = torch.permute(frames, (0, 3, 1, 2)).type(torch.float32)

        # cast uint8 to float32
        # TODO: check carefully how to do this cast
        # torchvision.transforms.ToTensor

        if self.transforms is not None:
            # consider the entire video as batch of images
            frames = self.transforms(frames)
        
        #return path_to_video, self.current_video, video_cls
        return frames, video_cls

class SpatioTemporalDataset(VideoDataset):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.avi', '.mp4'),
        num_clips: int = 16
        ) -> None:

        super().__init__(root, transforms, extensions)
        self.num_clips = num_clips

    def _sample_frames(
        self, 
        video: Tensor) -> Tensor:
        # divide a video into num_clip buckets, uniform sampling from each bucket.
        # sort of segment based sampling
        # video tensor is TxCxHxW
        # TODO: sampling for validation
        num_frames = video.shape[0]

        if self.num_clips > num_frames:
            raise ValueError('num_clips should be lower than frames in video')
        
        average_duration = video.shape[0] // self.num_clips
        offsets = np.multiply(list(range(self.num_clips)), average_duration) + np.random.randint(average_duration, size=self.num_clips)

        return torch.index_select(video, 0, torch.tensor(offsets, dtype=torch.int32))

    def __getitem__(
        self, 
        index: int) -> Tuple[Tensor, int]:
        video, label = super().__getitem__(index)
        sampled_video = self._sample_frames(video)

        return sampled_video, label

if __name__ == '__main__':
    # import torchvision.transforms as T
    # import utils as U
 
    # pipeline = T.Compose([
    #     T.RandomResizedCrop((112, 112)),
    #     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
 
    # dataset = SpatioTemporalDataset(
    #     '/mnt/d/serao/archive/Real Life Violence Dataset',
    #     transforms=pipeline,
    #     num_clips=16
    # )
 
    # print(f'Dataset size: {len(dataset)}')
 
    # train_set, val_set = U.stratified_random_split(dataset, (0.7, 0.3), dataset.targets)
 
    # print(f'Train size: {len(train_set)}')
    # print(f'Train type: {train_set}')
    # print(f'Validation size: {len(val_set)}')
    # print(f'Validation type: {val_set}')

    # dataloader returns a batch
    # we need to extract the videoclip
    # itr_dataset = DataLoader(dataset)

    # video, cls = next(iter(itr_dataset))
    # print(f'Video shape: {video.shape}')
    # print(f'Video dtype: {video.dtype}')
    # print(f'Video class: {cls}')
    
    from tqdm import tqdm

    def dataset_index(dataset):
        dataset_itr = iter(dataset)

        with open('./train_index.txt', 'w') as f:
            for path, video, label in tqdm(dataset_itr):
                #print(f'Path {path[0]}, Video {video.shape[1]}, Label {label.item()}')
                f.write(path[0] + ' ' + str(video.shape[1]) + ' ' + str(label.item()) + '\n')
    
    video_dataset = VideoDataset('/mnt/d/serao/fight_detection')
    dataset = DataLoader(video_dataset, num_workers=0)
    dataset_index(dataset)
import os
import torch

from torch import Tensor
from typing import Callable, Optional, Tuple
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

        self.targets = [video[1] for video in self.videos_index]
    
    def targets(self):
        return self.targets

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
        # batch of images -> Tensor[B, C, H, W]
        # B batch size, C channels
        video = torch.permute(frames, (0, 3, 1, 2))

        # cast uint8 to float32
        # TODO: check carefully how to do this cast
        # torchvision.transforms.ToTensor
        video = video.type(torch.float32)

        if self.transforms is not None:
            # consider the entire video as batch of images
            print(f'pre-transform: {video.shape}')
            video = self.transforms(video)
        
        print(f'post-transform: {video.shape}')

        return video, video_cls

if __name__ == '__main__':
    import torchvision.transforms as T
    import utils as U

    pipeline = T.Compose([
        T.RandomResizedCrop((112, 112)),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    dataset = VideoDataset(
        '/mnt/d/serao/archive/Real Life Violence Dataset',
        transforms=pipeline
    )

    print(f'Dataset size: {len(dataset)}')

    train_set, val_set = U.stratified_random_split(dataset, (0.7, 0.3), dataset.targets)

    print(f'Train size: {len(train_set)}')
    print(f'Train type: {train_set}')
    print(f'Validation size: {len(val_set)}')
    print(f'Validation type: {val_set}')

    # dataloader returns a batch
    # we need to extract the videoclip
    itr_dataset = DataLoader(dataset)

    video, cls = next(iter(itr_dataset))
    print(f'Video shape: {video.shape}')
    print(f'Video dtype: {video.dtype}')
    print(f'Video class: {cls}')
    
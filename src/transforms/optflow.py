import torch
import cv2
import numpy as np

from torchvision.transforms import Compose, Resize, CenterCrop

def resize(frame, size):
    transform = Compose([Resize(size), CenterCrop(size)])
    return transform(frame)

def opticalFlowDenseRLOF(frames, size):
    num_frames = frames.shape[0]

    if num_frames % 2 != 0:
        raise ValueError('number of frames should be a multiple of 2')

    # input frame should be a torch tensor TxCxHxW
    frames = resize(frames, size)
    # convert frames from TxCxHxW to TxHxWxC
    # required by opencv
    frames = torch.permute(frames, (0, 2, 3, 1))
    result = []

    for i in range(0, num_frames, 2):
        prev_frame = frames[i].numpy()
        next_frame = frames[i+1].numpy()

        # create HSV & make Value a constant
        hsv = np.zeros_like(prev_frame)
        hsv[..., 1] = 255

        # calculate optical flow
        flow = cv2.optflow.calcOpticalFlowDenseRLOF(prev_frame, next_frame, None)

        # encoding: convert the algorithm's output into polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # use Hue and Value to encode the optical flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # convert HSV image into RGB
        opt = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        opt = torch.from_numpy(opt)
        result.append(opt)
    
    return torch.permute(torch.stack(result), (0, 3, 1, 2))
    
    

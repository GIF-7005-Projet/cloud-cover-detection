import torch
import torch.nn.functional as F

def resize(input,
           size=None,
           scale_factor=None,
           mode='bilinear',
           align_corners=None):
    
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    
    return F.interpolate(input, size, scale_factor, mode, align_corners)

import torch.nn as nn


class DWConv(nn.Module):
    """
    Taken from: https://github.com/NVlabs/SegFormer/tree/master
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(in_channels=dim,
                                out_channels=dim, 
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=True,
                                groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

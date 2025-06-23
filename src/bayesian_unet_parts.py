import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# 0.1 best dropout
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.conv    = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(skip_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv    = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Pytorch applies softmax when using CrossLossEntropy

    def forward(self, x):
        return self.conv(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Applies two 2D convolutional layers, each followed by BatchNorm.
    then applies ReLu

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for both convolutional layers.
    """
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
        """Defines the forward pass of the DoubleConv block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        return self.double_conv(x)

class Down(nn.Module):
    """Downsampling block for the Bayesian U-Net encoder path.

    This block consists of a MaxPool2d layer for downsampling, a Dropout2d layer
    for regularization (Monte Carlo Dropout), and a DoubleConv block for feature
    extraction.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for the DoubleConv block.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.5.
    """
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.conv    = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        """Defines the forward pass of the Down block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor after max-pooling, dropout, and double convolution.
                          Shape: (B, out_channels, H/2, W/2).
        """
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    """Upsampling block for the U-Net decoder path.

    This block performs transposed convolution for upsampling, concatenates with
    a skip connection, and then applies a DoubleConv block.

    Args:
        in_channels (int): Number of input channels from the lower layer.
        skip_channels (int): Number of channels from the skip connection.
        out_channels (int): Number of output channels for the DoubleConv block.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(skip_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        """Defines the forward pass of the Up block.

        Args:
            x1 (torch.Tensor): Input tensor from the lower layer (upsampled).
                               Shape: (B, in_channels, H_low, W_low).
            x2 (torch.Tensor): Skip connection tensor from the encoder path.
                               Shape: (B, skip_channels, H_high, W_high).

        Returns:
            torch.Tensor: Output tensor after upsampling, concatenation, and double convolution.
                          Shape: (B, out_channels, H_high, W_high).
        """
        x1 = self.up(x1)

        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolutional layer for the Bayesian U-Net.

    This layer applies a 1x1 convolution to map the feature channels to the
    desired number of output classes.

    Args:
        in_channels (int): Number of input channels from the previous layer.
        out_channels (int): Number of output channels (number of classes).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv    = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Defines the forward pass of the OutConv block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
                          This typically represents the raw logits before softmax.
        """
        return self.conv(x)
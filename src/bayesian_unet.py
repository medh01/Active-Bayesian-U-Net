import torch
import torch.nn as nn
from bayesian_unet_parts import DoubleConv, Down, Up, OutConv

class BayesianUNet(nn.Module):
    """
    A Bayesian U-Net model for image segmentation.

    This model uses dropout layers in the encoder path to introduce uncertainty,
    which is a key feature of Bayesian neural networks. The architecture consists
    of an encoder (down-sampling path), a bottleneck, and a decoder (up-sampling path)
    with skip connections.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        features: list = [64, 128, 256, 512],
        dropout_prob: float = 0.5
    ):
        """
        Initializes the Bayesian U-Net model.

        Args:
            in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            out_channels (int): Number of output channels (number of classes).
            features (list): A list of integers specifying the number of features
                             at each level of the Bayesian U-Net.
            dropout_prob (float): The dropout probability to be used in the Down blocks.
        """
        super().__init__()

        # 1) Initial DoubleConv (no pooling yet)
        self.init_conv = DoubleConv(in_channels, features[0])  # 1 → 64

        # 2) Three Down blocks (producing skip1, skip2, skip3):
        self.down1 = Down(features[0], features[1], dropout_prob=dropout_prob)  # 64 → 128
        self.down2 = Down(features[1], features[2], dropout_prob=dropout_prob)  # 128 → 256
        self.down3 = Down(features[2], features[3], dropout_prob=dropout_prob)  # 256 → 512

        # 3) Bottleneck implemented as Down4 (512 → 1024)
        self.bottleneck = Down(features[3], features[3] * 2, dropout_prob=dropout_prob)

        # 4) Four Up blocks:
        #    Up1: (1024 from bottleneck) + skip3 (512) → output 512
        self.up1 = Up(in_channels=features[3] * 2, skip_channels=features[3], out_channels=features[3])
        #    Up2: (512 from up1) + skip2 (256) → output 256
        self.up2 = Up(in_channels=features[3], skip_channels=features[2], out_channels=features[2])
        #    Up3: (256 from up2) + skip1 (128) → output 128
        self.up3 = Up(in_channels=features[2], skip_channels=features[1], out_channels=features[1])
        #    Up4: (128 from up3) + skip0 (64 from init) → output 64
        self.up4 = Up(in_channels=features[1], skip_channels=features[0], out_channels=features[0])

        # 5) Final 1×1 conv → Softmax
        self.out_conv = OutConv(features[0], out_channels)

    def forward(self, x):
        """
        Defines the forward pass of the Bayesian U-Net.

        Args:
            x (torch.Tensor): The input tensor of shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: The output tensor (segmentation map) of shape (N, out_channels, H, W).
        """
        # Encoder
        x0 = self.init_conv(x)   # (N, 64,  H,   W)
        x1 = self.down1(x0)      # (N,128, H/2, W/2)
        x2 = self.down2(x1)      # (N,256, H/4, W/4)
        x3 = self.down3(x2)      # (N,512, H/8, W/8)
        x4 = self.bottleneck(x3) # (N,1024,H/16,W/16) ← bottleneck/down4

        # Decoder (concatenate with skips in reverse order)
        u1 = self.up1(x4, x3)    # (N,512, H/8, W/8)
        u2 = self.up2(u1, x2)    # (N,256, H/4, W/4)
        u3 = self.up3(u2, x1)    # (N,128, H/2, W/2)
        u4 = self.up4(u3, x0)    # (N, 64,  H,   W)

        # Final 1×1 conv + Softmax
        out = self.out_conv(u4)  # (N, out_channels, H, W)
        return out

# test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BayesianUNet().to(device)

    x = torch.randn(2, 1, 256, 256).to(device)
    preds = model(x)
    print("Output shape:", preds.shape)

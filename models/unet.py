import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    This is the core building block of the U-Net architecture.
    Use consecutive convolutional layers
    Each followed by batch normalization and ReLU activation
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        """
        nn.Conv2d:
        Applies a 2D convolution filter (kernel size 3×3)
        padding=1 ensures the output spatial size stays the same
        First conv changes input channels → output channels
        Second conv keeps it at out_channels

        nn.BatchNorm2d
        Normalizes activations across the batch and channels
        Helps stabilize and speed up training
        Reduces internal covariate shift
        """
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Final output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        # Bottleneck
        bn = self.bottleneck(self.pool4(d4))

        # Decoder
        up4 = self.up4(bn)
        dec4 = self.dec4(torch.cat([up4, d4], dim=1))

        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, d3], dim=1))

        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, d2], dim=1))

        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, d1], dim=1))

        # Output
        return self.out_conv(dec1)
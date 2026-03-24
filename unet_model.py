import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class Encoder2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels=16, dropout=0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = ConvBlock(16, 32, dropout=0.15)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = ConvBlock(32, 64, dropout=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = ConvBlock(64, 128, dropout=0.3)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        return p4, [c1, c2, c3, c4]


class Decoder2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upconv6 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(in_channels=256, out_channels=128, dropout=0.3)

        self.upconv7 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(in_channels=128, out_channels=64, dropout=0.2)

        self.upconv8 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=64, out_channels=32, dropout=0.15)

        self.upconv9 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.conv9 = ConvBlock(in_channels=32, out_channels=16, dropout=0.1)

    def forward(self, x, skips):
        u6 = self.upconv6(x)
        u6 = torch.cat([u6, skips[3]], dim=1)
        c6 = self.conv6(u6)

        u7 = self.upconv7(c6)
        u7 = torch.cat([u7, skips[2]], dim=1)
        c7 = self.conv7(u7)

        u8 = self.upconv8(c7)
        u8 = torch.cat([u8, skips[1]], dim=1)
        c8 = self.conv8(u8)

        u9 = self.upconv9(c8)
        u9 = torch.cat([u9, skips[0]], dim=1)
        c9 = self.conv9(u9)

        return c9

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = Encoder2D(in_channels)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256, dropout=0.3)
        self.decoder = Decoder2D(in_channels=256)
        self.out_conv = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        p4, skips = self.encoder(x)
        c5 = self.conv5(p4)
        c9 = self.decoder(c5, skips)
        outputs = self.out_conv(c9)
        return outputs
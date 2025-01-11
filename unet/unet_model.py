""" Full assembly of the parts to form the complete network """
import os
from torchvision import transforms

from .unet_parts import *

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))

        # self.cbam1 = CBAM(channel=64)
        # self.cbam2 = CBAM(channel=128)
        # self.cbam3 = CBAM(channel=256)
        # self.cbam4 = CBAM(channel=512)
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        # x1 = self.cbam1(x1) + x1
        x2 = self.down1(x1)
        # x2 = self.cbam2(x2) + x2
        x3 = self.down2(x2)
        # x3 = self.cbam3(x3) + x3
        x4 = self.down3(x3)
        # x4 = self.cbam4(x4) + x4
        x5 = self.down4(x4)
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        logits = self.outc(u4)
        # self.save_image(x1, 'x1')
        # self.save_image(x2, 'x2')
        # self.save_image(x3, 'x3')
        # self.save_image(x4, 'x4')
        # self.save_image(x5, 'x5')
        # self.save_image(u1, 'up1')
        # self.save_image(u2, 'up2')
        # self.save_image(u3, 'up3')
        # self.save_image(u4, 'up4')
        # self.save_image(logits, 'logits')
        return logits

    def save_image(self, tensor, name):
        os.makedirs('unet_test', exist_ok=True)
        # 将每个通道单独保存为灰度图像
        for i in range(tensor.shape[1]):
            channel = tensor[:, i, :, :].cpu().squeeze(0)
        transform = transforms.ToPILImage()
        img = transform(channel)
        img.save(os.path.join('unet_test', f'{name}_channel_{i}.png'))


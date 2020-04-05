# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes1, n_classes2):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 24)
        self.down1 = down(24, 48)
        self.down2 = down(48, 96)
        self.down3 = down(96, 384)
        self.down4 = down(384, 384)
        self.up1 = up(768, 192)
        self.up2 = up(288, 48)
        self.up3 = up(96, 48)
        self.up4 = up(72, 30)
        self.outc = outconv(30, n_classes1)

        self.inc2 = inconv(n_channels + n_classes1, 16)
        self.down5 = down(16, 32)
        self.down6 = down(32, 64)
        self.up5 = up(144, 32)
        self.up6 = up(72, 8)
        self.outc2 = outconv(8, n_classes2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up1(x5, x4)
        x6 = self.up2(x6, x3)
        x6 = self.up3(x6, x2)
        x6 = self.up4(x6, x1)
        x_rgb = self.outc(x6)

        x=torch.cat([x, x_rgb],dim=1)
        x1_2 = self.inc2(x)
        x2_2 = self.down5(x1_2)
        x3_2 = self.down6(x2_2)

        x6 = self.up5(x3_2, torch.cat([x2,x2_2], dim=1))
        x6 = self.up6(x6, torch.cat([x1,x1_2], dim=1))
        x_alpha = self.outc2(x6)

        x=torch.cat([x_rgb,x_alpha],dim=1)
        return x
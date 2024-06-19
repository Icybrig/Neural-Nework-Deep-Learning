import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32 * upscale_factor ** 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.PixelShuffle(upscale_factor=upscale_factor)
        )

    def forward(self, x):
        return self.seq(x)

from torch import Tensor, nn
import einops
from torch.nn import Conv2d, BatchNorm2d, ReLU


class CNNBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3):
        super().__init__()
        self.conv1 = Conv2d(channels_in, channels_out, kernel_size=kernel_size, padding='same')
        self.bn1 = BatchNorm2d(channels_out)
        self.relu = ReLU()
        self.conv2 = Conv2d(channels_out, channels_out, kernel_size=kernel_size, padding='same')
        self.bn2 = BatchNorm2d(channels_out)
        self.channels = (channels_in, channels_out)

    def forward(self, x: Tensor):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        return x


class CNN(nn.Module):
    def __init__(self, channels, repeats, kernel_size=5):
        super().__init__()
        self.out_channels = channels[-1]

        self.seq = nn.ModuleList()
        self.res = nn.ModuleList()
        for i in range(len(channels) - 1):
            layer = nn.Sequential()
            for r in range(repeats):
                c_in = channels[i + 1] if r != 0 else channels[i]
                c_out = channels[i + 1]
                layer.append(CNNBlock(c_in, c_out, kernel_size))
            self.seq.append(layer)
            self.res.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def res_reduction_factor(self):
        return 2 ** (max(0, len(self.seq) - 1))

    def forward(self, x: Tensor):
        x = einops.rearrange(x, '... l w c -> ... c l w')
        for i in range(len(self.seq)):
            x = self.seq[i](x) + self.res[i](x)
            if i != len(self.seq) - 1:
                x = self.pool(x)
        x = einops.rearrange(x, ' ... c l w -> ... l w c')
        return x

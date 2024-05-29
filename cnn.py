from torch import Tensor, nn
import einops
from torch.nn import Conv1d, Conv2d, BatchNorm2d, ReLU


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


class FlatConv(nn.Module):
    def __init__(self, channels_in, channels_out, width):
        super().__init__()
        self.conv1 = Conv1d(width * channels_in, width * channels_in, kernel_size=3, groups=width, padding='same')
        self.conv2 = Conv1d(width * channels_in, width * channels_out, kernel_size=5, groups=width, padding='same')

    def forward(self, x: Tensor):
        w = x.shape[-1]
        x = einops.rearrange(x, '... c l w -> ... (w c) l')
        x = self.conv1(x)
        x = self.conv2(x)
        x = einops.rearrange(x, '... (w c) l -> ... c l w', w=w)
        return x


class CNN(nn.Module):
    def __init__(self, channels, repeats, width, kernel_size=5):
        super().__init__()
        self.out_channels = channels[-1]

        self.flat_conv = FlatConv(channels[0], channels[1], width)
        self.seq = nn.ModuleList()
        self.res = nn.ModuleList()
        for i in range(len(channels) - 1):
            layer = nn.Sequential()
            for r in range(repeats):
                c_in = channels[i + 1] if r != 0 or i == 0 else channels[i]
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
            res = self.res[i](x)
            if i == 0:
                x = self.flat_conv(x)
            x = self.seq[i](x) + res
            if i != len(self.seq) - 1:
                x = self.pool(x)
        x = einops.rearrange(x, ' ... c l w -> ... l w c')
        return x


if __name__ == '__main__':
    import torch
    print('Running FlatConv test')
    cv = FlatConv(3, 1, 64)
    try:
        import matplotlib.pyplot as plt
        import torchinfo

        torchinfo.summary(cv)
    except:
        pass
    def gen():
        b, c, l, w = 32, 3, 128, 64
        x = torch.rand((c, l, w * b))
        sine = einops.rearrange(torch.sin(0.1 * torch.arange(0, c * l)), '(c l) -> c l 1', c=c, l=l)
        x[..., torch.rand(b * w) > 0.5] = sine
        x = einops.rearrange(x, 'c l (b w) -> b c l w', b=b, c=c)
        ans = x.clone()
        ans[:, 0, 1:, :] -= 0.5 * ans[:, 1, :-1, :]
        return x, ans[:, 0:1, :, :]
    optimizer = torch.optim.Adam({*cv.parameters()}, 0.0002)
    losses = [0.0]
    for its in range(1, 100000 + 1):
        optimizer.zero_grad()
        x, ans = gen()
        out = cv(x)
        loss: Tensor = torch.nn.functional.mse_loss(out, ans)
        loss.backward()
        optimizer.step()
        losses[-1] += loss.item() / 50.0
        if its % 50 == 0:
            try:
                print(losses[-1])
                plt.title(f'Loss: {losses[-1]}')
                plt.plot(list(range(len(losses))), losses)
                plt.ylim(0, 0.25)
                plt.xlim(0, (len(losses) + 110) // 100 * 100)
                plt.show()
            except:
                pass
            losses += [0.0]

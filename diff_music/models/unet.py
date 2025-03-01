from torch import Tensor, nn
import torch

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.01),
            nn.GroupNorm(32, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)
    
class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.GroupNorm(32, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
    
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            Downsample(in_channels, out_channels),
            ResBlock(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(in_channels),
            Upsample(in_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class UnetDenoiser(nn.Module):
    def __init__(self, in_channels: int, start_channels: int, channels_mult: list[int]):
        super().__init__()
        self.num_blocks = len(channels_mult) - 1
        channels = [start_channels * m for m in channels_mult]
        
        self.in_block = nn.Sequential(
            nn.Conv2d(in_channels + 1, channels[0], 3, padding=1),
            nn.LeakyReLU(0.01),
            nn.GroupNorm(32, channels[0]),
            ResBlock(channels[0]),
        )
        self.downs = nn.ModuleList()
        for i in range(self.num_blocks):
            self.downs.append(
                DownBlock(channels[i], channels[i+1]),
            )

        self.ups = nn.ModuleList()
        for i in range(self.num_blocks):
            self.ups.append(
                UpBlock(channels[i+1] * 2 if i != self.num_blocks - 1 else channels[i+1], channels[i]),
            )

        self.out_block = nn.Sequential(
            ResBlock(channels[0] * 2),
            nn.GroupNorm(32, channels[0]* 2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(channels[0]* 2, in_channels, 3, padding=1),
        )
        

    def forward(self, x: Tensor, time_steps: Tensor) -> Tensor:
        '''
        x: (b, c, h, w)
        t: (b)
        '''

        time_steps = time_steps.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, time_steps], dim=1)
        x = self.in_block(x) # (b, start_channels, h, w)

        skip_connections = []
        for down in self.downs:
            skip_connections.append(x)
            x = down(x)
            
        for up in reversed(self.ups):
            x = up(x)
            x = torch.cat([x, skip_connections.pop()], dim=1)

        x = self.out_block(x)
        
        return x

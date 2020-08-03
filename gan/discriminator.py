from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ms = 16
        self.t1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=ms, kernel_size=(4 , 4), stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t2 = nn.Sequential(
            nn.Conv2d(in_channels=ms, out_channels=ms*2, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(ms*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t3 = nn.Sequential(
            nn.Conv2d(in_channels=ms*2, out_channels=ms*4, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(ms*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t4 = nn.Sequential(
            nn.Conv2d(in_channels=ms*4, out_channels=ms*8, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(ms*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t5 = nn.Sequential(
            nn.Conv2d(in_channels=ms*8, out_channels=ms*16, kernel_size=(3, 3), stride=2, padding=0),
            nn.BatchNorm2d(ms*16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t6 = nn.Sequential(
            nn.Conv2d(in_channels=ms*16, out_channels=1, kernel_size=(3, 3), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.t5(x)
        x = self.t6(x)
        return x  # output of discriminator
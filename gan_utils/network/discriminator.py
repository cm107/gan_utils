import torch.nn as nn
import torch.optim as optim
from .util import weights_init

class Discriminator(nn.Module):
    def __init__(
        self, ngpu: int=1,
        nc: int=3, ndf: int=64
    ):
        """
        ngpu - number of GPUs available. If this is 0, code will run in CPU mode. If this number is greater than 0 it will run on that number of GPUs
        nc - number of color channels in the input images. For color images this is 3
        ndf - sets the depth of feature maps propagated through the discriminator
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def init_weights(self):
        # for child in self.main.children():
        #     weights_init(child)
        self.apply(weights_init) # Equivalent to above

    def get_optim(self, lr: float=0.0002, beta1: float=0.5) -> optim.Adam:
        return optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))
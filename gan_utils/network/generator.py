import torch.nn as nn
import torch.optim as optim
from .util import weights_init

class Generator(nn.Module):
    def __init__(
        self, ngpu: int=1,
        nc: int=3, nz: int=100, ngf: int=64
    ):
        """
        ngpu - number of GPUs available. If this is 0, code will run in CPU mode. If this number is greater than 0 it will run on that number of GPUs
        nc - number of color channels in the input images. For color images this is 3
        nz - length of latent vector
        ngf - relates to the depth of feature maps carried through the generator
        """
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    def init_weights(self):
        # for child in self.main.children():
        #     weights_init(child)
        self.apply(weights_init) # Equivalent to above
    
    def get_optim(self, lr: float=0.0002, beta1: float=0.5) -> optim.Adam:
        return optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))
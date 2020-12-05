import torch
import torch.nn as nn

# custom weights initialization called on netG and netD
def weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_device(ngpu: int=1) -> torch.device:
    return torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def get_criterion() -> nn.BCELoss:
    return nn.BCELoss()

def get_noise(device: torch.device, batch_size: int=64, nz: int=100) -> torch.Tensor:
    return torch.randn(batch_size, nz, 1, 1, device=device)

def get_fixed_noise(device: torch.device, nz: int=100) -> torch.Tensor:
    """Create batch of latent vectors that we will use to visualize
    the progression of the generator

    Args:
        device (torch.device): device task is run on
        nz (int, optional): length of latent vector.
            Defaults to 100.
    """
    return get_noise(device=device, batch_size=64, nz=nz)
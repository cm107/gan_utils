import cv2
import random
import torch.nn as nn
import torch
import torchvision.utils as vutils
from tqdm import tqdm

from ..network.generator import Generator
from ..network.discriminator import Discriminator
from ..network.util import get_device, get_criterion, get_noise, get_fixed_noise
from ..dataset.util import get_dataset, get_dataloader
from ..network import constants

class Trainer:
    def __init__(
        self, dataroot: str, image_size: int=64,
        batch_size: int=128, num_workers: int=2,
        num_epochs: int=5,
        ngpu: int=1,
        nc: int=3, nz: int=100, ngf: int=64, ndf: int=64,
        lr: float=0.0002, beta1: float=0.5,
        seed: int=None
    ):
        """
        ngpu - number of GPUs available. If this is 0, code will run in CPU mode. If this number is greater than 0 it will run on that number of GPUs
        nc - number of color channels in the input images. For color images this is 3
        nz - length of latent vector
        ngf - relates to the depth of feature maps carried through the generator
        ndf - sets the depth of feature maps propagated through the discriminator
        """
        # Seed
        self._set_seed(seed=seed)

        # Initialize Networks
        self.device = get_device(ngpu=ngpu)
        self.netG = Generator(ngpu=ngpu, nc=nc, nz=nz, ngf=ngf).to(self.device)
        self.netD = Discriminator(ngpu=ngpu, nc=nc, ndf=ndf).to(self.device)
        if (self.device.type == 'cuda') and (ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(ngpu)))
            self.netD = nn.DataParallel(self.netD, list(range(ngpu)))
        self.netG.init_weights()
        self.netD.init_weights()
        self.optimizerG = self.netG.get_optim(lr=lr, beta1=beta1)
        self.optimizerD = self.netD.get_optim(lr=lr, beta1=beta1)
        
        # Initialize Dataset
        self.dataset = get_dataset(dataroot=dataroot, image_size=image_size)
        self.dataloader = get_dataloader(
            dataset=self.dataset, batch_size=batch_size,
            num_workers=num_workers, shuffle=True
        )
        self.num_epochs = num_epochs

        # Other
        self.criterion = get_criterion()
        self.fixed_noise = get_fixed_noise(device=self.device, nz=nz)
        self.nz = nz

    def _set_seed(self, seed: int):
        if seed is None:
            seed0 = random.randint(1, 10000)
            random.seed(seed0)
            torch.manual_seed(seed0)
            print(f'Random Seed: {seed0}')
        else:
            random.seed(seed)
            torch.manual_seed(seed)
            print(f'Manual Seed: {seed}')

    def train(self, show_pbar: bool=True):
        # Training Loop
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        pbar = tqdm(total=self.num_epochs*len(self.dataloader)) if show_pbar else None
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                if pbar is not None:
                    pbar.set_description(f'epoch[{epoch}/{self.num_epochs}], batch[{i}/{len(self.dataloader)}]')

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), constants.real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = get_noise(device=self.device, batch_size=b_size, nz=self.nz)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(constants.fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(constants.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 10 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, self.num_epochs, i, len(self.dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 10 == 0) or ((epoch == self.num_epochs-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    img_grid = vutils.make_grid(fake, padding=2, normalize=True)
                    img_grid = img_grid.numpy().transpose((1,2,0)) * 255
                    img_grid = cv2.cvtColor(img_grid, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('vis.png', img_grid)

                iters += 1
                if pbar is not None:
                    pbar.update()
        if pbar is not None:
            pbar.close()

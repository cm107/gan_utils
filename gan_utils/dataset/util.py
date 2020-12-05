from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as T

def get_dataset(dataroot: str, image_size: int=64) -> dset.ImageFolder:
    """Gets the dataset used for GAN training.

    Args:
        dataroot (str): Root directory that contains all dataset folders.
            Note: This is not the image folder itself, but the parent directory
            of all image folders.
        image_size (int, optional): The size of the image cropped
            from the center of each image in the dataset.
            Defaults to 64.

    Returns:
        dset.ImageFolder
    """
    transforms = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    return dset.ImageFolder(
        root=dataroot,
        transform=transforms
    )

def get_dataloader(
    dataset: dset.ImageFolder, batch_size: int=128,
    num_workers: int=2, shuffle: bool=True
):
    return DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers
    )

def preview_data(dataroot: str):
    from ..network.util import get_device
    import numpy as np
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt

    dataset = get_dataset(dataroot=dataroot, image_size=64)
    dataloader = get_dataloader(
        dataset=dataset, batch_size=128, num_workers=2,
        shuffle=True
    )
    device = get_device(ngpu=1)
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

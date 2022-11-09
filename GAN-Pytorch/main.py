import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms as transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.nets import Generator, Discriminator

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of sepochs")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["mnist", "cifar10"],
    default="mnist",
    help="dataset for training",
)
parser.add_argument("--latent_dim", type=int, default=100, help="dimension of noise z")
parser.add_argument("--expr_name", type=str, default="baseline", help="name of the experiment")
opt = parser.parse_args()
print("Running Options: \n", opt)

# Tensorboard
writer = SummaryWriter(f"runs/{opt.dataset}/{opt.expr_name}")


def train():
    running_loss_G = 0.0
    running_loss_D = 0.0

    with tqdm(total=opt.n_epochs) as pbar:
        for epoch in range(opt.n_epochs):
            for i, (real_imgs, _) in enumerate(train_loader):
                if device == "cuda":
                    real_imgs = real_imgs.cuda()

                # Target
                real = torch.ones(real_imgs.size(0), 1, device=device, dtype=torch.float32)
                fake = torch.zeros(real_imgs.size(0), 1, device=device, dtype=torch.float32)

                # Noise(z) sampling from normal distribution
                z = torch.tensor(
                    np.random.normal(0, 1, size=(real_imgs.size(0), opt.latent_dim)),
                    device=device,
                    dtype=torch.float32,
                )

                # Generate imgs
                fake_imgs = generator(z)

                # ** Iteration for discriminator **
                optimizer_D.zero_grad()
                real_loss = loss(discriminator(real_imgs), real)  # -log(D(x)
                fake_loss = loss(discriminator(fake_imgs.detach()), fake)  # -log(1-D(G(z)))
                loss_D = real_loss + fake_loss  # -(log(D(x)) + log(1-D(G(z))))
                loss_D.backward()
                optimizer_D.step()

                #  ** Iteration for generator **
                optimizer_G.zero_grad()
                loss_G = loss(discriminator(fake_imgs), real)  # -log(D(G(z)))
                loss_G.backward()
                optimizer_G.step()

                running_loss_G += loss_G.item()
                running_loss_D += loss_D.item()

                # tqdm
                pbar.set_postfix(
                    {
                        "Generator loss": "%.3f" % loss_G.item(),
                        "Discriminator loss": "%.3f" % loss_D.item(),
                    }
                )

                # TODO Tensorboard, logging
                if i % 10 == 9:  # every 10 minibatch
                    # log = f"Epoch: {epoch}/{opt.n_epochs}, loss_G: {running_loss_G/10}, loss_D: {running_loss_D/10}"
                    # print(log)

                    writer.add_scalar(
                        "Generator loss", running_loss_G / 10, epoch * len(train_loader) + i
                    )
                    writer.add_scalar(
                        "Discriminator loss", running_loss_D / 10, epoch * len(train_loader) + i
                    )

                    running_loss_G = 0.0
                    running_loss_D = 0.0

            scheduler_D.step()
            scheduler_G.step()
            pbar.update(1)

            # Save fake images
            fake_img_dir = os.path.join("./images", opt.dataset)
            os.makedirs(fake_img_dir, exist_ok=True)
            save_image(
                fake_imgs.data[:25], f"{fake_img_dir}/epoch_{epoch}.png", nrow=5, normalize=True
            )


if __name__ == "__main__":
    # Set deviceshape and texture paramet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    ## Define dataset & dataloader
    dataset_name = opt.dataset
    data_root = os.path.join("./data", dataset_name)
    os.makedirs(data_root, exist_ok=True)

    if dataset_name == "mnist":
        img_dims = [1, 28, 28]
        mean = 0.5
        std = 0.5
    else:
        img_dims = [3, 32, 32]
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.247, 0.243, 0.261)

    data_transforms = transforms.Compose(
        [
            transforms.Resize(img_dims[-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if dataset_name == "mnist":
        train_dataset = MNIST(root=data_root, train=True, download=True, transform=data_transforms)
        valid_dataset = MNIST(root=data_root, train=False, download=True, transform=data_transforms)
    else:
        train_dataset = CIFAR10(
            root=data_root, train=True, download=True, transform=data_transforms
        )
        valid_dataset = CIFAR10(
            root=data_root, train=False, download=True, transform=data_transforms
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
    )

    # Define model, loss, optimizer
    generator = Generator(opt.latent_dim, img_dims)
    discriminator = Discriminator(img_dims)
    loss = nn.BCELoss()

    if device == "cuda":
        generator.cuda()
        discriminator.cuda()
        loss.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.9)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.9)

    train()

import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, img_dims):
        super(Generator, self).__init__()
        self.img_dims = img_dims

        # Define layers
        def _block(in_dim, out_dim, normalize=True):
            block = [nn.Linear(in_dim, out_dim)]
            if normalize:
                block.append(nn.BatchNorm1d(out_dim))
            block.append(nn.LeakyReLU())
            return block

        self.model = nn.Sequential(
            *_block(latent_dim, 128, False),
            *_block(128, 256),
            *_block(256, 512),
            *_block(512, 1024),
            nn.Linear(1024, np.prod(img_dims)),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.model(z)
        img = x.view(x.size(0), *self.img_dims)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_dims):
        super(Discriminator, self).__init__()
        # Define layers
        def _block(in_dim, out_dim, p=0.1):
            block = [nn.Linear(in_dim, out_dim)]
            if p > 0:
                block.append(nn.LeakyReLU())
                block.append(nn.Dropout(p))
            return block

        self.model = nn.Sequential(
            *_block(np.prod(img_dims), 256),
            *_block(256, 64),
            *_block(64, 32),
            *_block(32, 1, p=0),
            nn.Sigmoid(),
        )

    def forward(self, img):
        # Define forward pass
        x = img.view(img.size(0), -1)
        x = self.model(x)
        return x

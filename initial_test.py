import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


batch_size = 64
image_size = 64
nz = 100
num_epochs = 200
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root="fish_dataset", transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)


G = Generator(nz).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        

        D.zero_grad()
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        output = D(real_images).view(-1, 1)
        loss_real = criterion(output, real_labels)
        loss_real.backward()
        
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = G(noise)
        output = D(fake_images.detach()).view(-1, 1)
        loss_fake = criterion(output, fake_labels)
        loss_fake.backward()
        optimizerD.step()
        
        G.zero_grad()
        output = D(fake_images).view(-1, 1)
        loss_G = criterion(output, real_labels)
        loss_G.backward()
        optimizerG.step()
        
        if i % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], D Loss: {loss_real.item() + loss_fake.item()}, G Loss: {loss_G.item()}")
    

    with torch.no_grad():
        fake_samples = G(fixed_noise).cpu()
        vutils.save_image(fake_samples, f"output/epoch_{epoch}.png", normalize=True)

def generate_fish_images(num_images=10):
    noise = torch.randn(num_images, nz, 1, 1, device=device)
    fake_images = G(noise).cpu()
    for i, img in enumerate(fake_images):
        vutils.save_image(img, f"generated/fish_{i}.png", normalize=True)
    print(f"Generated {num_images} fish images!")

generate_fish_images()

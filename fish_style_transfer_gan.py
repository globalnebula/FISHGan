import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from ptflops import get_model_complexity_info

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# Generator Network
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, features=64):
        super(Generator, self).__init__()
        
        # Initial downsampling
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        # Downsampling blocks
        self.down1 = self._block(features, features*2, 4, 2, 1)
        self.down2 = self._block(features*2, features*4, 4, 2, 1)
        self.down3 = self._block(features*4, features*8, 4, 2, 1)
        self.down4 = self._block(features*8, features*8, 4, 2, 1)
        self.down5 = self._block(features*8, features*8, 4, 2, 1)
        self.down6 = self._block(features*8, features*8, 4, 2, 1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1),
            nn.ReLU()
        )
        
        # Upsampling blocks
        self.up1 = self._block(features*8, features*8, 4, 2, 1, True)
        self.up2 = self._block(features*8*2, features*8, 4, 2, 1, True)
        self.up3 = self._block(features*8*2, features*8, 4, 2, 1, True)
        self.up4 = self._block(features*8*2, features*8, 4, 2, 1, True)
        self.up5 = self._block(features*8*2, features*4, 4, 2, 1, True)
        self.up6 = self._block(features*4*2, features*2, 4, 2, 1, True)
        self.up7 = self._block(features*2*2, features, 4, 2, 1, True)
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, output_channels, 4, 2, 1),
            nn.Tanh()
        )
        
        # Profile model complexity
        macs, params = get_model_complexity_info(self, (3, 256, 256), as_strings=False, print_per_layer_stat=False)
        print(f'Generator GMACs: {macs/1e9:.2f}')
        print(f'Generator Parameters: {params/1e6:.2f}M')
        
        # Triplet loss
        self.triplet_loss = TripletLoss()
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, up=False):
        if up:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode="reflect"),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
    
    def forward(self, x, positive=None, negative=None):
        # Initial downsampling
        x = self.initial(x)
        
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        # Bottleneck
        bottleneck = self.bottleneck(d6)
        
        # Compute triplet loss if positive and negative samples are provided
        loss_ht = 0
        if positive is not None and negative is not None:
            # Get bottleneck features for positive and negative samples
            pos_bottleneck = self.bottleneck(self.down6(self.down5(self.down4(self.down3(self.down2(self.down1(self.initial(positive))))))))
            neg_bottleneck = self.bottleneck(self.down6(self.down5(self.down4(self.down3(self.down2(self.down1(self.initial(negative))))))))
            
            # Compute triplet loss
            loss_ht = self.triplet_loss(bottleneck, pos_bottleneck, neg_bottleneck)
        
        # Upsampling
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d6], 1))
        u3 = self.up3(torch.cat([u2, d5], 1))
        u4 = self.up4(torch.cat([u3, d4], 1))
        u5 = self.up5(torch.cat([u4, d3], 1))
        u6 = self.up6(torch.cat([u5, d2], 1))
        u7 = self.up7(torch.cat([u6, d1], 1))
        
        # Final layer
        output = self.final(torch.cat([u7, x], 1))
        
        return output, loss_ht

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        
        layers = []
        in_channels = in_channels
        
        for feature in features:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, 4, 2, 1, padding_mode="reflect"),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = feature
        
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, 1, 4, 1, 1, padding_mode="reflect"),
                nn.Sigmoid()
            )
        )
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Custom Dataset for SeaCLEF
class FishDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Training function
def train_gan(
    generator_A2B, generator_B2A,
    discriminator_A, discriminator_B,
    dataloader_A, dataloader_B,
    num_epochs=100,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Loss functions
    adversarial_loss = nn.BCELoss()
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()
    
    # Optimizers
    optimizer_G = optim.Adam(
        list(generator_A2B.parameters()) + list(generator_B2A.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(discriminator_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(discriminator_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Training loop
    for epoch in range(num_epochs):
        for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), 1, 16, 16), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), 1, 16, 16), requires_grad=False).to(device)
            
            # Train Generators
            optimizer_G.zero_grad()
            
            # Identity loss
            loss_id_A = identity_loss(generator_B2A(real_A)[0], real_A)
            loss_id_B = identity_loss(generator_A2B(real_B)[0], real_B)
            
            # GAN loss
            fake_B, loss_ht_A2B = generator_A2B(real_A, real_B, real_A)  # real_B as positive, real_A as negative
            loss_GAN_A2B = adversarial_loss(discriminator_B(fake_B), valid)
            
            fake_A, loss_ht_B2A = generator_B2A(real_B, real_A, real_B)  # real_A as positive, real_B as negative
            loss_GAN_B2A = adversarial_loss(discriminator_A(fake_A), valid)
            
            # Cycle loss
            reconstructed_A = generator_B2A(fake_B)[0]
            loss_cycle_A = cycle_loss(reconstructed_A, real_A)
            
            reconstructed_B = generator_A2B(fake_A)[0]
            loss_cycle_B = cycle_loss(reconstructed_B, real_B)
            
            # Total generator loss
            loss_G = (
                loss_GAN_A2B + loss_GAN_B2A +
                loss_cycle_A * 10.0 + loss_cycle_B * 10.0 +
                loss_id_A * 5.0 + loss_id_B * 5.0 +
                loss_ht_A2B * 2.0 + loss_ht_B2A * 2.0  # Add triplet losses with weight 2.0
            )
            
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator A
            optimizer_D_A.zero_grad()
            loss_real = adversarial_loss(discriminator_A(real_A), valid)
            loss_fake = adversarial_loss(discriminator_A(fake_A.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # Train Discriminator B
            optimizer_D_B.zero_grad()
            loss_real = adversarial_loss(discriminator_B(real_B), valid)
            loss_fake = adversarial_loss(discriminator_B(fake_B.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
            
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader_A)}] "
                    f"[D loss: {loss_D_A.item() + loss_D_B.item():.4f}] "
                    f"[G loss: {loss_G.item():.4f}] "
                    f"[HT loss: {(loss_ht_A2B.item() + loss_ht_B2A.item()):.4f}]"
                )
                
                # Save generated images
                save_image(fake_B, f"generated_images/fake_B_{epoch}_{i}.png")
                save_image(fake_A, f"generated_images/fake_A_{epoch}_{i}.png")

# Main function
def main():
    # Create directories
    os.makedirs("generated_images", exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create datasets and dataloaders
    dataset_A = FishDataset("datasets/fishA/trainA", transform=transform)
    dataset_B = FishDataset("datasets/fishB/trainB", transform=transform)
    
    dataloader_A = DataLoader(dataset_A, batch_size=1, shuffle=True)
    dataloader_B = DataLoader(dataset_B, batch_size=1, shuffle=True)
    
    # Initialize models
    generator_A2B = Generator().to(device)
    generator_B2A = Generator().to(device)
    discriminator_A = Discriminator().to(device)
    discriminator_B = Discriminator().to(device)
    
    # Train the GAN
    train_gan(
        generator_A2B, generator_B2A,
        discriminator_A, discriminator_B,
        dataloader_A, dataloader_B,
        num_epochs=100,
        device=device
    )

if __name__ == "__main__":
    main() 
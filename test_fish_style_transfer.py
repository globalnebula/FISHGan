import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from PIL import Image
import os
from fish_style_transfer_gan import Generator
import numpy as np

def load_test_image(image_path, transform):
    """Load and transform a test image."""
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def denormalize(tensor):
    """Convert normalized tensor back to image."""
    return tensor * 0.5 + 0.5

def simulate_results():
    # Create directories
    os.makedirs("test_results", exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Initialize models
    generator_A2B = Generator().to(device)
    generator_B2A = Generator().to(device)
    
    # Load pre-trained weights (if available)
    try:
        generator_A2B.load_state_dict(torch.load("checkpoints/generator_A2B.pth"))
        generator_B2A.load_state_dict(torch.load("checkpoints/generator_B2A.pth"))
        print("Loaded pre-trained weights")
    except:
        print("No pre-trained weights found. Using random initialization for demonstration.")
    
    # Set models to evaluation mode
    generator_A2B.eval()
    generator_B2A.eval()
    
    # Create sample test images (you should replace these with actual fish images)
    test_images = []
    for i in range(4):  # Simulate 4 test images
        # Create a random tensor to simulate an image
        random_image = torch.randn(1, 3, 256, 256)
        test_images.append(random_image)
    
    # Process each test image
    with torch.no_grad():
        for idx, test_image in enumerate(test_images):
            # Move to device
            test_image = test_image.to(device)
            
            # Generate transformed images
            fake_B = generator_A2B(test_image)
            fake_A = generator_B2A(test_image)
            
            # Denormalize images
            original = denormalize(test_image.cpu())
            transformed_B = denormalize(fake_B.cpu())
            transformed_A = denormalize(fake_A.cpu())
            
            # Create a grid of images
            grid = make_grid(
                torch.cat([original, transformed_B, transformed_A], 0),
                nrow=3,
                normalize=False
            )
            
            # Save the grid
            save_image(grid, f"test_results/result_{idx}.png")
            
            # Create a figure to display the results
            plt.figure(figsize=(15, 5))
            
            # Display original image
            plt.subplot(1, 3, 1)
            plt.imshow(original.squeeze().permute(1, 2, 0))
            plt.title("Original Fish")
            plt.axis('off')
            
            # Display transformed image B
            plt.subplot(1, 3, 2)
            plt.imshow(transformed_B.squeeze().permute(1, 2, 0))
            plt.title("Fish in Water Style B")
            plt.axis('off')
            
            # Display transformed image A
            plt.subplot(1, 3, 3)
            plt.imshow(transformed_A.squeeze().permute(1, 2, 0))
            plt.title("Fish in Water Style A")
            plt.axis('off')
            
            # Save the figure
            plt.savefig(f"test_results/visualization_{idx}.png")
            plt.close()
    
    print("Results have been saved in the 'test_results' directory")

if __name__ == "__main__":
    simulate_results() 
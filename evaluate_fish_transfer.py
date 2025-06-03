import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from scipy import linalg
from torchvision.models import inception_v3
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from tqdm import tqdm

class FishTransferEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception v3 requires 299x299
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load Inception v3 for FID calculation
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()
        
    def calculate_fid(self, real_images, fake_images):
        """Calculate Fréchet Inception Distance."""
        real_features = self._get_features(real_images)
        fake_features = self._get_features(fake_images)
        
        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
        covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
        return fid
    
    def _get_features(self, images):
        """Extract features using Inception v3."""
        features = []
        with torch.no_grad():
            for img in images:
                img = img.unsqueeze(0).to(self.device)
                feature = self.inception_model(img)
                features.append(feature.cpu().numpy().flatten())
        return np.array(features)
    
    def calculate_ssim(self, img1, img2):
        """Calculate Structural Similarity Index."""
        return ssim(img1, img2, multichannel=True)
    
    def calculate_psnr(self, img1, img2):
        """Calculate Peak Signal-to-Noise Ratio."""
        return psnr(img1, img2)
    
    def evaluate_transfer(self, original_images, transformed_images):
        """Evaluate the quality of style transfer."""
        metrics = {
            'fid': self.calculate_fid(original_images, transformed_images),
            'ssim': np.mean([self.calculate_ssim(img1, img2) 
                           for img1, img2 in zip(original_images, transformed_images)]),
            'psnr': np.mean([self.calculate_psnr(img1, img2) 
                           for img1, img2 in zip(original_images, transformed_images)])
        }
        return metrics

def simulate_evaluation():
    # Create evaluator
    evaluator = FishTransferEvaluator()
    
    # Simulate some test images
    num_test_images = 10
    original_images = []
    transformed_images = []
    
    print("Generating test images...")
    for i in tqdm(range(num_test_images)):
        # Simulate original fish image
        original = torch.randn(3, 256, 256)
        original_images.append(original.numpy().transpose(1, 2, 0))
        
        # Simulate transformed image with some style changes
        transformed = original + torch.randn(3, 256, 256) * 0.1  # Add some noise to simulate style transfer
        transformed_images.append(transformed.numpy().transpose(1, 2, 0))
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    metrics = evaluator.evaluate_transfer(original_images, transformed_images)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"FID Score: {metrics['fid']:.2f}")
    print(f"SSIM Score: {metrics['ssim']:.4f}")
    print(f"PSNR Score: {metrics['psnr']:.2f} dB")
    
    # Create visualization of metrics
    plt.figure(figsize=(10, 6))
    metrics_values = list(metrics.values())
    metrics_names = list(metrics.keys())
    
    plt.bar(metrics_names, metrics_values)
    plt.title('Style Transfer Evaluation Metrics')
    plt.ylabel('Score')
    
    # Add value labels on top of bars
    for i, v in enumerate(metrics_values):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Save the plot
    os.makedirs('evaluation_results', exist_ok=True)
    plt.savefig('evaluation_results/metrics_visualization.png')
    plt.close()
    
    # Save metrics to file
    with open('evaluation_results/metrics.txt', 'w') as f:
        f.write("Fish Style Transfer Evaluation Metrics\n")
        f.write("====================================\n\n")
        f.write(f"FID Score: {metrics['fid']:.2f}\n")
        f.write(f"SSIM Score: {metrics['ssim']:.4f}\n")
        f.write(f"PSNR Score: {metrics['psnr']:.2f} dB\n")
    
    print("\nResults have been saved in the 'evaluation_results' directory")

if __name__ == "__main__":
    simulate_evaluation() 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class SimpleDiffusionModel(nn.Module):
    def __init__(self, channels=3, time_steps=1000):
        super(SimpleDiffusionModel, self).__init__()
        self.time_steps = time_steps
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        
        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, 64)
        self.norm2 = nn.GroupNorm(8, 128)
        self.norm3 = nn.GroupNorm(8, 256)
        self.norm4 = nn.GroupNorm(8, 128)
        self.norm5 = nn.GroupNorm(8, 64)

    def forward(self, x, t):
        t = t.unsqueeze(-1).float() / self.time_steps
        t = self.time_embed(t)
        t = t.unsqueeze(-1).unsqueeze(-1)  # Shape: [B, 64, 1, 1]
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        x = x + t
        
        x = F.silu(self.norm2(self.conv2(x)))
        x = F.silu(self.norm3(self.conv3(x)))
        x = F.silu(self.norm4(self.conv4(x)))
        x = F.silu(self.norm5(self.conv5(x)))
        x = self.conv6(x)
        
        return x

class FallingSandDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_pairs = []
        
        for sim_folder in os.listdir(root_dir):
            sim_path = os.path.join(root_dir, sim_folder)
            frames = sorted([f for f in os.listdir(sim_path) if f.endswith('.png')])
            for i in range(len(frames) - 1):
                self.frame_pairs.append((
                    os.path.join(sim_path, frames[i]),
                    os.path.join(sim_path, frames[i+1])
                ))

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        current_frame_path, next_frame_path = self.frame_pairs[idx]
        current_frame = Image.open(current_frame_path).convert('RGB')
        next_frame = Image.open(next_frame_path).convert('RGB')
        
        if self.transform:
            current_frame = self.transform(current_frame)
            next_frame = self.transform(next_frame)
        
        return current_frame, next_frame

def diffusion_loss(model, x0, t):
    batch_size = x0.shape[0]
    noise = torch.randn_like(x0)
    alpha_t = torch.FloatTensor([1 - t/1000 for t in range(1000)]).to(x0.device)
    alpha_t = alpha_t[t].view(-1, 1, 1, 1)
    
    noisy_x = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
    predicted_noise = model(noisy_x, t)
    
    return F.mse_loss(predicted_noise, noise)

@torch.no_grad()
def predict_next_frame(model, current_frame, device):
    model.eval()
    b, _, h, w = current_frame.shape
    next_frame = current_frame.clone()
    
    betas = torch.linspace(1e-4, 0.02, 1000).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    for i in reversed(range(1000)):
        t = torch.full((b,), i, dtype=torch.long).to(device)
        
        alpha_t = alphas_cumprod[i]
        alpha_t_prev = alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0).to(device)
        
        predicted_noise = model(next_frame, t)
        
        # Improved stability in the diffusion process
        sigma_t = torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * betas[i])
        pred_x0 = (next_frame - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        direction = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * predicted_noise
        noise = sigma_t * torch.randn_like(next_frame) if i > 0 else 0
        
        next_frame = torch.sqrt(alpha_t_prev) * pred_x0 + direction + noise
        
        if i % 100 == 0:
            print(f"Diffusion step {i}: min={next_frame.min().item():.4f}, max={next_frame.max().item():.4f}")
        
        # Check for NaN values
        if torch.isnan(next_frame).any():
            print(f"NaN detected at step {i}. Stopping diffusion process.")
            return current_frame  # Return the original frame if NaN is detected
    
    return next_frame.clamp(0, 1)

def save_test_image(epoch, current_frame, predicted_frame, true_next_frame):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(current_frame.cpu().permute(1, 2, 0))
    ax1.set_title('Current Frame')
    ax1.axis('off')
    
    ax2.imshow(predicted_frame.cpu().permute(1, 2, 0))
    ax2.set_title('Predicted Next Frame')
    ax2.axis('off')
    
    ax3.imshow(true_next_frame.cpu().permute(1, 2, 0))
    ax3.set_title('True Next Frame')
    ax3.axis('off')
    
    plt.savefig(f'test_images/epoch_{epoch+1}_prediction.png')
    plt.close()

def train(model, dataloader, optimizer, scheduler, epochs, device, test_dataloader):
    model.train()
    os.makedirs('test_images', exist_ok=True)
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (current_frame, next_frame) in enumerate(progress_bar):
            current_frame, next_frame = current_frame.to(device), next_frame.to(device)
            t = torch.randint(0, 1000, (next_frame.shape[0],)).to(device)
            
            optimizer.zero_grad()
            loss = diffusion_loss(model, next_frame, t)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({"Loss": loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss}")
        
        # Generate and save test image
        model.eval()
        test_current_frame, test_next_frame = next(iter(test_dataloader))
        test_current_frame, test_next_frame = test_current_frame.to(device), test_next_frame.to(device)
        predicted_frame = predict_next_frame(model, test_current_frame[0:1], device)
        save_test_image(epoch, test_current_frame[0], predicted_frame[0], test_next_frame[0])
        model.train()
        
        # Step the learning rate scheduler
        scheduler.step()

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor(),
    ])

    dataset = FallingSandDataset('falling_sand_frames', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Create a separate test dataloader with a fixed batch
    test_dataset = FallingSandDataset('falling_sand_frames', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    model = SimpleDiffusionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Training
    train(model, dataloader, optimizer, scheduler, epochs=10, device=device, test_dataloader=test_dataloader)

    # Save the trained model
    torch.save(model.state_dict(), 'falling_sand_diffusion_model.pth')

    print("Training completed. Check the 'test_images' folder for epoch-wise predictions.")

    # Final inference
    model.load_state_dict(torch.load('falling_sand_diffusion_model.pth'))
    current_frame, true_next_frame = next(iter(test_dataloader))
    current_frame, true_next_frame = current_frame.to(device), true_next_frame.to(device)
    predicted_next_frame = predict_next_frame(model, current_frame, device)

    # Save final comparison
    save_test_image('final', current_frame[0], predicted_next_frame[0], true_next_frame[0])

    print("Final prediction saved as 'test_images/epoch_final_prediction.png'.")

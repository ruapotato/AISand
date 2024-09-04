import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

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

def train(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (current_frame, next_frame) in enumerate(dataloader):
            current_frame, next_frame = current_frame.cuda(), next_frame.cuda()
            t = torch.randint(0, 1000, (next_frame.shape[0],)).cuda()
            
            optimizer.zero_grad()
            loss = diffusion_loss(model, next_frame, t)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss}")

@torch.no_grad()
def predict_next_frame(model, current_frame):
    model.eval()
    device = next(model.parameters()).device
    b, _, h, w = current_frame.shape
    next_frame = current_frame.clone()
    
    for i in reversed(range(1000)):
        t = torch.full((b,), i, dtype=torch.long).to(device)
        noise = model(next_frame, t)
        alpha_t = 1 - i / 1000
        next_frame = (next_frame - torch.sqrt(1 - alpha_t) * noise) / torch.sqrt(alpha_t)
        if i > 1:
            noise = torch.randn_like(next_frame) * torch.sqrt((1/1000) * (1000-i) / (1000-i+1))
            next_frame = next_frame + noise
    
    return next_frame.clamp(0, 1)

if __name__ == "__main__":
    # Setup
    transform = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor(),
    ])

    dataset = FallingSandDataset('falling_sand_frames', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = SimpleDiffusionModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training
    train(model, dataloader, optimizer, epochs=10)

    # Save the trained model
    torch.save(model.state_dict(), 'falling_sand_diffusion_model.pth')

    # Inference
    model.load_state_dict(torch.load('falling_sand_diffusion_model.pth'))
    current_frame, _ = next(iter(dataloader))
    current_frame = current_frame.cuda()
    predicted_next_frame = predict_next_frame(model, current_frame[0:1])

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(current_frame[0].cpu().permute(1, 2, 0))
    ax1.set_title('Current Frame')
    ax1.axis('off')
    ax2.imshow(predicted_next_frame[0].cpu().permute(1, 2, 0))
    ax2.set_title('Predicted Next Frame')
    ax2.axis('off')
    plt.savefig('frame_prediction_comparison.png')
    plt.close()

    print("Training and inference completed. Check 'frame_prediction_comparison.png' for results.")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
WIDTH, HEIGHT = 255, 255
EMPTY, SAND, WATER, PLANT, WOOD, ACID, FIRE = range(7)

COLORS = {
    EMPTY: (0, 0, 0),
    SAND: (194, 178, 128),
    WATER: (0, 0, 255),
    PLANT: (0, 255, 0),
    WOOD: (139, 69, 19),
    ACID: (255, 255, 0),
    FIRE: (255, 0, 0)
}

COLOR_TO_CELL = {tuple(v): k for k, v in COLORS.items()}

class SandDataset(Dataset):
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
        current_frame = self.image_to_grid(Image.open(current_frame_path))
        next_frame = self.image_to_grid(Image.open(next_frame_path))
        
        if self.transform:
            current_frame = self.transform(current_frame)
            next_frame = self.transform(next_frame)
        
        return current_frame, next_frame

    def image_to_grid(self, image):
        np_image = np.array(image)
        grid = np.zeros((WIDTH, HEIGHT), dtype=np.int64)
        for cell_type, color in COLORS.items():
            grid[(np_image == color).all(axis=2)] = cell_type
        return grid

class CellularAutomatonCNN(nn.Module):
    def __init__(self):
        super(CellularAutomatonCNN, self).__init__()
        self.conv1 = nn.Conv2d(7, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 7, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        return self.conv4(x)

def train_model(model, train_loader, val_loader, num_epochs=10, device="cuda"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for current_frame, next_frame in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            current_frame, next_frame = current_frame.to(device), next_frame.to(device)
            current_frame = torch.nn.functional.one_hot(current_frame.long(), num_classes=7).permute(0, 3, 1, 2).float()

            optimizer.zero_grad()
            output = model(current_frame)
            loss = criterion(output, next_frame)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for current_frame, next_frame in val_loader:
                current_frame, next_frame = current_frame.to(device), next_frame.to(device)
                current_frame = torch.nn.functional.one_hot(current_frame.long(), num_classes=7).permute(0, 3, 1, 2).float()
                output = model(current_frame)
                loss = criterion(output, next_frame)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_sand_model.pth')

    return train_losses, val_losses

def predict_next_frame(model, current_frame, device="cuda"):
    model.eval()
    with torch.no_grad():
        current_frame = torch.nn.functional.one_hot(current_frame.long(), num_classes=7).permute(0, 3, 1, 2).float().to(device)
        output = model(current_frame)
        return torch.argmax(output, dim=1)

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig('loss_plot.png')
    plt.close()

def visualize_prediction(input_frame, true_next_frame, predicted_frame):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(input_frame)
    ax1.set_title('Input Frame')
    ax1.axis('off')
    
    ax2.imshow(true_next_frame)
    ax2.set_title('True Next Frame')
    ax2.axis('off')
    
    ax3.imshow(predicted_frame)
    ax3.set_title('Predicted Next Frame')
    ax3.axis('off')
    
    plt.savefig('prediction_comparison.png')
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    dataset = SandDataset('falling_sand_frames')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model Training
    model = CellularAutomatonCNN()
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=50, device=device)

    # Plot Losses
    plot_losses(train_losses, val_losses)

    # Load Best Model
    model.load_state_dict(torch.load('best_sand_model.pth'))

    # Inference
    test_input, test_true_next = next(iter(val_loader))
    test_input = test_input.to(device)
    predicted_next = predict_next_frame(model, test_input, device)

    # Visualize Results
    input_frame = test_input[0].cpu().numpy()
    true_next_frame = test_true_next[0].cpu().numpy()
    predicted_frame = predicted_next[0].cpu().numpy()

    visualize_prediction(input_frame, true_next_frame, predicted_frame)

    print("Training completed. Check 'loss_plot.png' for training progress and 'prediction_comparison.png' for a sample prediction.")

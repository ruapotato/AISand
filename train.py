import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import logging
import os
import glob
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
EMPTY, SAND, WATER, PLANT, WOOD, ACID, FIRE, STEAM, SALT, TNT, WAX, OIL, LAVA, STONE = range(14)
COLORS = {
    EMPTY: (0, 0, 0),
    SAND: (194, 178, 128),
    WATER: (0, 0, 255),
    PLANT: (0, 255, 0),
    WOOD: (139, 69, 19),
    ACID: (255, 255, 0),
    FIRE: (255, 0, 0),
    STEAM: (200, 200, 200),
    SALT: (255, 255, 255),
    TNT: (255, 0, 128),
    WAX: (255, 220, 180),
    OIL: (80, 80, 80),
    LAVA: (255, 69, 0),
    STONE: (128, 128, 128)
}

class CellularAutomatonDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.frame_pairs = []
        self.preprocess_dataset()

    def preprocess_dataset(self):
        logger.info(f"Preprocessing dataset from {self.root_dir}...")
        simulation_dirs = sorted(glob.glob(os.path.join(self.root_dir, "simulation_*")))
        
        for sim_dir in tqdm(simulation_dirs, desc="Processing simulations"):
            frames = sorted(glob.glob(os.path.join(sim_dir, "frame_*.png")))
            for i in range(len(frames) - 1):
                self.frame_pairs.append((frames[i], frames[i+1]))
        
        logger.info(f"Preprocessing complete. Found {len(self.frame_pairs)} frame pairs.")

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        current_frame_path, next_frame_path = self.frame_pairs[idx]
        
        try:
            current_frame = self.image_to_grid(Image.open(current_frame_path))
            next_frame = self.image_to_grid(Image.open(next_frame_path))
            return current_frame, next_frame
        except Exception as e:
            logger.warning(f"Error processing images: {str(e)}")
            logger.warning(f"Skipping problematic pair: {current_frame_path} and {next_frame_path}")
            return self.__getitem__((idx + 1) % len(self))

    def image_to_grid(self, image):
        np_image = np.array(image)
        if np_image.shape[2] != 4:
            raise ValueError(f"Expected RGBA image, got shape {np_image.shape}")
        
        grid = np.zeros((np_image.shape[0], np_image.shape[1], len(COLORS)), dtype=np.float32)
        for i, color in enumerate(COLORS.values()):
            grid[:,:,i] = (np_image[:,:,0] == color[0]) & (np_image[:,:,1] == color[1]) & (np_image[:,:,2] == color[2])
        return torch.from_numpy(grid)

class ImprovedSandModel(nn.Module):
    def __init__(self, input_channels=14, hidden_size=64):
        super(ImprovedSandModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_size * 2)
        self.conv3 = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        return self.conv4(x)

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for current_frame, next_frame in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            current_frame = current_frame.permute(0, 3, 1, 2).float().to(device)
            next_frame = next_frame.argmax(dim=3).long().to(device)
            
            optimizer.zero_grad()
            outputs = model(current_frame)
            
            loss = criterion(outputs, next_frame)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for current_frame, next_frame in val_loader:
                current_frame = current_frame.permute(0, 3, 1, 2).float().to(device)
                next_frame = next_frame.argmax(dim=3).long().to(device)
                outputs = model(current_frame)
                loss = criterion(outputs, next_frame)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if epoch == 0 or val_loss < min(val_losses[:-1]):
            torch.save(model.state_dict(), 'best_improved_sand_model.pth')
            logger.info("Saved new best model")
    
    return train_losses, val_losses

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = CellularAutomatonDataset("enhanced_falling_sand_frames")
    
    if len(dataset) == 0:
        logger.error("No valid images found in the dataset. Please check your image files and paths.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    model = ImprovedSandModel(input_channels=14, hidden_size=64).to(device)
    
    num_epochs = 50
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, device)
    
    logger.info(f"Training complete. Final training loss: {train_losses[-1]:.4f}, Final validation loss: {val_losses[-1]:.4f}")

if __name__ == "__main__":
    main()

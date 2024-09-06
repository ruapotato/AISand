import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import logging
import os
import glob
from PIL import Image
import psutil
import heapq

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
    def __init__(self, root_dir, sequence_length=3, top_percent=0.25):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.top_percent = top_percent
        self.frame_sequences = []
        self._preprocess_dataset()

    def _preprocess_dataset(self):
        logger.info(f"Indexing dataset from {self.root_dir}...")
        simulation_dirs = sorted(glob.glob(os.path.join(self.root_dir, "simulation_*")))
        
        all_sequences = []
        for sim_dir in tqdm(simulation_dirs, desc="Processing simulations"):
            frames = sorted(glob.glob(os.path.join(sim_dir, "frame_*.png")))
            for i in range(len(frames) - self.sequence_length):
                sequence = frames[i:i+self.sequence_length+1]
                movement = self._calculate_movement(sequence)
                all_sequences.append((movement, sequence))
        
        # Sort sequences by movement (descending order) and select top 25%
        all_sequences.sort(reverse=True)
        num_selected = int(len(all_sequences) * self.top_percent)
        self.frame_sequences = [seq for _, seq in all_sequences[:num_selected]]
        
        logger.info(f"Indexing complete. Selected {len(self.frame_sequences)} frame sequences with highest movement.")

    def _calculate_movement(self, sequence):
        total_diff = 0
        for i in range(len(sequence) - 1):
            frame1 = np.array(Image.open(sequence[i]))
            frame2 = np.array(Image.open(sequence[i+1]))
            diff = np.sum(frame1 != frame2) / (frame1.shape[0] * frame1.shape[1] * frame1.shape[2])
            total_diff += diff
        return total_diff / (len(sequence) - 1)

    def __len__(self):
        return len(self.frame_sequences)

    def __getitem__(self, idx):
        frame_paths = self.frame_sequences[idx]
        
        frames = [self.image_to_grid(Image.open(path)) for path in frame_paths]
        input_sequence = np.stack(frames[:-1])
        target_frame = frames[-1]
        return torch.from_numpy(input_sequence).float(), torch.from_numpy(target_frame).float()

    def image_to_grid(self, image):
        np_image = np.array(image)
        grid = np.zeros((np_image.shape[0], np_image.shape[1], len(COLORS)), dtype=np.float32)
        for i, color in enumerate(COLORS.values()):
            grid[:,:,i] = (np_image[:,:,0] == color[0]) & (np_image[:,:,1] == color[1]) & (np_image[:,:,2] == color[2])
        return grid

class SimpleSandModel(nn.Module):
    def __init__(self, input_channels=14, hidden_size=32):
        super(SimpleSandModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: (batch, sequence, height, width, channels)
        batch_size, seq_len, height, width, channels = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # (batch, sequence, channels, height, width)
        x = x.view(batch_size * seq_len, channels, height, width)
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        
        x = x.view(batch_size, seq_len, channels, height, width)
        return x[:, -1]  # Return only the last frame prediction, shape: (batch, channels, height, width)

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for input_sequence, target_frame in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_sequence = input_sequence.to(device)
            target_frame = target_frame.argmax(dim=3).long().to(device)
            
            optimizer.zero_grad()
            outputs = model(input_sequence)
            
            loss = criterion(outputs, target_frame)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for input_sequence, target_frame in val_loader:
                input_sequence = input_sequence.to(device)
                target_frame = target_frame.argmax(dim=3).long().to(device)
                outputs = model(input_sequence)
                loss = criterion(outputs, target_frame)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Memory Usage: {psutil.virtual_memory().percent}%")
        
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 'best_sand_model.pth')
            logger.info("Saved new best model")
            best_val_loss = val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = CellularAutomatonDataset("enhanced_falling_sand_frames", sequence_length=3, top_percent=0.25)
    
    if len(dataset) == 0:
        logger.error("No valid images found in the dataset. Please check your image files and paths.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SimpleSandModel(input_channels=14, hidden_size=32).to(device)
    
    num_epochs = 50
    train_model(model, train_loader, val_loader, num_epochs, device)
    
    logger.info("Training complete.")

if __name__ == "__main__":
    main()

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
import random

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

ELEMENT_NAMES = ["EMPTY", "SAND", "WATER", "PLANT", "WOOD", "ACID", "FIRE", "STEAM", "SALT", "TNT", "WAX", "OIL", "LAVA", "STONE"]

class CurriculumAutomatonDataset(Dataset):
    def __init__(self, root_dir, sequence_length=3, elements=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.elements = elements
        self.frame_sequences = []
        self._preprocess_dataset()

    def _preprocess_dataset(self):
        logger.info(f"Indexing dataset from {self.root_dir}...")
        scenario_dirs = glob.glob(os.path.join(self.root_dir, "*"))
        
        for scenario_dir in tqdm(scenario_dirs, desc="Processing scenarios"):
            scenario_name = os.path.basename(scenario_dir)
            if self.elements is not None:
                if 'all_elements' in self.elements:
                    if scenario_name.startswith("all_elements_instance_"):
                        self._process_all_elements_instance(scenario_dir)
                elif scenario_name.startswith("single_"):
                    element_color = scenario_name[7:]
                    if element_color not in [str(COLORS[e]) for e in self.elements if e != 'all_elements']:
                        continue
                elif scenario_name.startswith("pair_"):
                    element_colors = scenario_name[5:].split('_')
                    if not any(color in [str(COLORS[e]) for e in self.elements if e != 'all_elements'] for color in element_colors):
                        continue
                else:
                    continue
            
            simulation_dirs = sorted(glob.glob(os.path.join(scenario_dir, "simulation_*")))
            self._process_simulation_dirs(simulation_dirs)
        
        logger.info(f"Indexing complete. Total frame sequences: {len(self.frame_sequences)}")

    def _process_all_elements_instance(self, instance_dir):
        all_elements_dir = os.path.join(instance_dir, "all_elements")
        if os.path.exists(all_elements_dir):
            simulation_dirs = sorted(glob.glob(os.path.join(all_elements_dir, "simulation_*")))
            self._process_simulation_dirs(simulation_dirs)

    def _process_simulation_dirs(self, simulation_dirs):
        for sim_dir in simulation_dirs:
            frames = sorted(glob.glob(os.path.join(sim_dir, "frame_*.png")))
            for i in range(len(frames) - self.sequence_length):
                sequence = frames[i:i+self.sequence_length+1]
                self.frame_sequences.append(sequence)

    def __len__(self):
        return len(self.frame_sequences)

    def __getitem__(self, idx):
        frame_paths = self.frame_sequences[idx]
        
        frames = []
        for path in frame_paths:
            frame = self.image_to_grid(Image.open(path))
            frames.append(frame)
        
        input_sequence = np.stack(frames[:-1])
        target_frame = frames[-1]
        
        return torch.from_numpy(input_sequence).long(), torch.from_numpy(target_frame).long()

    def image_to_grid(self, image):
        np_image = np.array(image)
        grid = np.zeros((np_image.shape[0], np_image.shape[1]), dtype=np.int64)
        for i, color in enumerate(COLORS.values()):
            mask = (np_image[:,:,0] == color[0]) & (np_image[:,:,1] == color[1]) & (np_image[:,:,2] == color[2])
            grid[mask] = i
        return grid

class EnhancedSandModel(nn.Module):
    def __init__(self, input_channels=14, hidden_size=64):
        super(EnhancedSandModel, self).__init__()
        self.embedding = nn.Embedding(input_channels, input_channels)
        self.conv1 = nn.Conv2d(input_channels * 3, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_size, input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, seq_len, height, width = x.shape
        x = self.embedding(x)
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(batch_size, seq_len * x.shape[2], height, width)
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)
        
        return x

def calculate_element_losses(outputs, targets):
    element_losses = []
    for i in range(len(ELEMENT_NAMES)):
        element_mask = (targets == i)
        if element_mask.sum() > 0:
            element_loss = nn.functional.cross_entropy(outputs[element_mask], targets[element_mask])
            element_losses.append(element_loss.item())
        else:
            element_losses.append(0)
    return element_losses

def train_model_curriculum(model, root_dir, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    
    elements = list(range(1, len(ELEMENT_NAMES)))  # Exclude EMPTY
    random.shuffle(elements)
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        if epoch < num_epochs // 3:
            focus_elements = elements[:len(elements)//3]
        elif epoch < num_epochs * 2 // 3:
            focus_elements = elements[:len(elements)*2//3]
        else:
            focus_elements = elements + ["all_elements"]
        
        logger.info(f"Focusing on elements: {[ELEMENT_NAMES[e] if isinstance(e, int) else e for e in focus_elements]}")
        
        dataset = CurriculumAutomatonDataset(root_dir, sequence_length=3, elements=focus_elements)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        model.train()
        train_loss = 0.0
        train_element_losses = [0.0] * len(ELEMENT_NAMES)
        
        for input_sequence, target_frame in tqdm(train_loader, desc="Training"):
            input_sequence = input_sequence.to(device)
            target_frame = target_frame.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_sequence)
            
            outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, outputs.shape[1])
            target_frame = target_frame.view(-1)
            
            loss = criterion(outputs, target_frame)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            element_losses = calculate_element_losses(outputs, target_frame)
            train_element_losses = [tl + el for tl, el in zip(train_element_losses, element_losses)]
        
        train_loss /= len(train_loader)
        train_element_losses = [el / len(train_loader) for el in train_element_losses]
        
        model.eval()
        val_loss = 0.0
        val_element_losses = [0.0] * len(ELEMENT_NAMES)
        
        with torch.no_grad():
            for input_sequence, target_frame in val_loader:
                input_sequence = input_sequence.to(device)
                target_frame = target_frame.to(device)
                outputs = model(input_sequence)
                
                outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, outputs.shape[1])
                target_frame = target_frame.view(-1)
                
                loss = criterion(outputs, target_frame)
                val_loss += loss.item()
                element_losses = calculate_element_losses(outputs, target_frame)
                val_element_losses = [vl + el for vl, el in zip(val_element_losses, element_losses)]
        
        val_loss /= len(val_loader)
        val_element_losses = [el / len(val_loader) for el in val_element_losses]
        
        scheduler.step(val_loss)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Memory Usage: {psutil.virtual_memory().percent}%")
        
        sorted_elements = sorted(zip(ELEMENT_NAMES, val_element_losses), key=lambda x: x[1], reverse=True)
        logger.info("Top 5 problematic elements:")
        for element, loss in sorted_elements[:5]:
            logger.info(f"{element}: {loss:.4f}")
        
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 'best_curriculum_sand_model.pth')
            logger.info("Saved new best model")
            best_val_loss = val_loss
        
        elements = [e for _, e in sorted(zip(val_element_losses, range(len(ELEMENT_NAMES))), reverse=True)]
        elements = [e for e in elements if e != EMPTY]  # Keep EMPTY excluded

def test_element(model, element_folder, device):
    test_dataset = CurriculumAutomatonDataset(element_folder, sequence_length=3)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_sequence, target_frame in test_loader:
            input_sequence = input_sequence.to(device)
            target_frame = target_frame.to(device)
            outputs = model(input_sequence)
            loss = nn.functional.cross_entropy(outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1]), target_frame.reshape(-1))
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += target_frame.numel()
            correct += (predicted == target_frame).sum().item()
    
    accuracy = 100.0 * correct / total
    test_loss /= len(test_loader)
    
    return test_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = EnhancedSandModel(input_channels=14, hidden_size=64).to(device)
    
    num_epochs = 20  # Shortened to 20 epochs
    train_model_curriculum(model, "curriculum_falling_sand_frames", num_epochs, device)
    
    logger.info("Training complete. Testing individual elements and all elements...")

    # Test individual elements
    for element_name, color in COLORS.items():
        element_folder = f"curriculum_falling_sand_frames/single_{color}"
        if os.path.exists(element_folder):
            test_loss, accuracy = test_element(model, element_folder, device)
            logger.info(f"{ELEMENT_NAMES[element_name]} - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        else:
            logger.warning(f"Folder not found for {ELEMENT_NAMES[element_name]}")

    # Test all elements
    all_elements_folder = "curriculum_falling_sand_frames/all_elements_instance_0/all_elements"
    if os.path.exists(all_elements_folder):
        test_loss, accuracy = test_element(model, all_elements_folder, device)
        logger.info(f"All Elements - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    else:
        logger.warning("Folder not found for All Elements")

    logger.info("All tests complete.")

if __name__ == "__main__":
    main()

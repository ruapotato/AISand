import pygame
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 255, 255
CELL_SIZE = 1
GRID_WIDTH, GRID_HEIGHT = WIDTH, HEIGHT
SEQUENCE_LENGTH = 3

# Colors and element types
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

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Falling Sand Game (New Model Prediction)")

# Create the grid and frame history
grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
frame_history = deque([grid.copy() for _ in range(SEQUENCE_LENGTH)], maxlen=SEQUENCE_LENGTH)

# Define the new model architecture
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

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedSandModel(input_channels=14, hidden_size=64).to(device)
model.load_state_dict(torch.load('best_curriculum_sand_model.pth', map_location=device, weights_only=True))
model.eval()

@torch.no_grad()
def predict_next_frame(model, frame_history):
    # Rotate the grids 90 degrees clockwise before prediction
    rotated_frames = [np.rot90(frame, k=-1) for frame in frame_history]
    
    # Stack the frames and convert to tensor
    input_sequence = np.stack(rotated_frames)
    input_sequence = torch.from_numpy(input_sequence).unsqueeze(0).long().to(device)
    
    output = model(input_sequence)
    output = output.permute(0, 2, 3, 1).contiguous().view(-1, output.shape[1])
    output = torch.softmax(output, dim=1)
    output = output.view(GRID_WIDTH, GRID_HEIGHT, -1)
    
    # Rotate the output back 90 degrees counterclockwise
    output = np.rot90(output.cpu().numpy(), k=1)
    return output

def draw_particles(surface, grid):
    # Create a 3D array of zeros with uint8 dtype
    color_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.uint8)
    
    # Use numpy's where function to set colors based on grid values
    for i, color in COLORS.items():
        mask = (grid == i)
        color_grid[mask] = color
    
    pygame.surfarray.blit_array(surface, color_grid)

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

running = True
current_element = SAND
brush_size = 5
clock = pygame.time.Clock()
surface = pygame.Surface((WIDTH, HEIGHT))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                x, y = pygame.mouse.get_pos()
                mask = create_circular_mask(GRID_HEIGHT, GRID_WIDTH, (y, x), brush_size)
                grid[mask] = current_element
                frame_history[-1] = grid.copy()  # Update the last frame in history
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:
                x, y = pygame.mouse.get_pos()
                mask = create_circular_mask(GRID_HEIGHT, GRID_WIDTH, (y, x), brush_size)
                grid[mask] = current_element
                frame_history[-1] = grid.copy()  # Update the last frame in history
        elif event.type == pygame.MOUSEWHEEL:
            brush_size = max(1, min(20, brush_size + event.y))
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                current_element = SAND
            elif event.key == pygame.K_2:
                current_element = WATER
            elif event.key == pygame.K_3:
                current_element = PLANT
            elif event.key == pygame.K_4:
                current_element = WOOD
            elif event.key == pygame.K_5:
                current_element = ACID
            elif event.key == pygame.K_6:
                current_element = FIRE
            elif event.key == pygame.K_7:
                current_element = STEAM
            elif event.key == pygame.K_8:
                current_element = SALT
            elif event.key == pygame.K_9:
                current_element = TNT
            elif event.key == pygame.K_0:
                current_element = WAX
            elif event.key == pygame.K_o:
                current_element = OIL
            elif event.key == pygame.K_l:
                current_element = LAVA
            elif event.key == pygame.K_s:
                current_element = STONE
            elif event.key == pygame.K_c:
                grid.fill(EMPTY)
                for i in range(SEQUENCE_LENGTH):
                    frame_history[i] = grid.copy()

    # Update grid using the model
    new_grid_probs = predict_next_frame(model, frame_history)
    new_grid = np.argmax(new_grid_probs, axis=-1)
    
    # Update grid (no blending, just use the new predictions)
    grid = new_grid
    
    # Update frame history
    frame_history.append(grid.copy())
    
    draw_particles(surface, grid)
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    
    fps = clock.get_fps()
    pygame.display.set_caption(f"FPS: {fps:.2f} - Brush Size: {brush_size}")
    clock.tick(60)  # Cap at 60 FPS
    
pygame.quit()

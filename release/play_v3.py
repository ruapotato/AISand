import pygame
import numpy as np
import torch
import torch.nn as nn

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 255, 255
CELL_SIZE = 1
GRID_WIDTH, GRID_HEIGHT = WIDTH, HEIGHT

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

# Create the grid
grid = np.zeros((GRID_WIDTH, GRID_HEIGHT, 14), dtype=np.float32)
grid[:,:,EMPTY] = 1  # Initialize with all cells empty

# Define the new model architecture
class SimpleSandModel(nn.Module):
    def __init__(self, input_channels=14, hidden_size=32):
        super(SimpleSandModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleSandModel(input_channels=14, hidden_size=32).to(device)
model.load_state_dict(torch.load('best_sand_model_v3.pth', map_location=device))
model.eval()

@torch.no_grad()
def predict_next_frame(model, current_frame):
    # Rotate the grid 90 degrees clockwise before prediction
    current_frame = np.rot90(current_frame, k=-1, axes=(0, 1)).copy()
    current_frame = current_frame.transpose(2, 0, 1)
    current_frame = torch.from_numpy(current_frame).unsqueeze(0).float().to(device)
    output = model(current_frame)
    # Rotate the output back 90 degrees counterclockwise
    output = torch.rot90(output.squeeze(), k=1, dims=(1, 2))
    output = torch.softmax(output, dim=0).cpu().numpy()
    # Ensure the output shape matches the input shape
    return output.transpose(1, 2, 0)

def draw_particles(surface, grid):
    color_grid = np.zeros((GRID_WIDTH, GRID_HEIGHT, 3), dtype=np.uint8)
    for i, color in COLORS.items():
        color_grid += (grid[:,:,i, np.newaxis] * color).astype(np.uint8)
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
                mask = create_circular_mask(GRID_WIDTH, GRID_HEIGHT, (y, x), brush_size)
                grid[mask, :] = 0
                grid[mask, current_element] = 1
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:
                x, y = pygame.mouse.get_pos()
                mask = create_circular_mask(GRID_WIDTH, GRID_HEIGHT, (y, x), brush_size)
                grid[mask, :] = 0
                grid[mask, current_element] = 1
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
                grid.fill(0)
                grid[:,:,EMPTY] = 1

    # Update grid using the model
    new_grid = predict_next_frame(model, grid)
    
    # Blend the new grid with the old grid to reduce flickering
    grid = grid * 0.7 + new_grid * 0.3
    
    draw_particles(surface, grid)
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    
    fps = clock.get_fps()
    pygame.display.set_caption(f"FPS: {fps:.2f} - Brush Size: {brush_size}")
    clock.tick()
    
pygame.quit()

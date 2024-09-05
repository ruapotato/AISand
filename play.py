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
pygame.display.set_caption("Enhanced Falling Sand Game (Model Prediction) - Diagonally Reflected")

# Create the grid
grid = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=np.int32)

# Define the model architecture (updated for 14 elements)
class CellularAutomatonCNN(nn.Module):
    def __init__(self):
        super(CellularAutomatonCNN, self).__init__()
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 14, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        return self.conv4(x)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CellularAutomatonCNN().to(device)
model.load_state_dict(torch.load('best_enhanced_sand_model.pth', map_location=device))
model.eval()

@torch.no_grad()
def predict_next_frame(model, current_frame):
    # Transpose the grid before prediction
    current_frame = current_frame.T
    current_frame = torch.from_numpy(current_frame).unsqueeze(0).to(device)
    current_frame = torch.nn.functional.one_hot(current_frame.long(), num_classes=14).permute(0, 3, 1, 2).float()
    output = model(current_frame)
    # Transpose the output back
    return torch.argmax(output, dim=1).squeeze().cpu().numpy().T

def draw_particles(surface, grid):
    # Create a 3D array of colors based on the grid
    color_map = np.array([COLORS[i] for i in range(14)])
    pixels = color_map[grid]
    
    # Update the surface with the pixel array
    pygame.surfarray.blit_array(surface, pixels)

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def reflect_coordinates(x, y):
    # Reflect over the diagonal from top-right to bottom-left
    return y, x

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
                x, y = reflect_coordinates(x, y)
                mask = create_circular_mask(GRID_WIDTH, GRID_HEIGHT, (x, y), brush_size)
                grid[mask] = current_element
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:
                x, y = pygame.mouse.get_pos()
                x, y = reflect_coordinates(x, y)
                mask = create_circular_mask(GRID_WIDTH, GRID_HEIGHT, (x, y), brush_size)
                grid[mask] = current_element
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

    # Update grid using the model
    grid = predict_next_frame(model, grid)
    draw_particles(surface, grid)
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    
    fps = clock.get_fps()
    pygame.display.set_caption(f"FPS: {fps:.2f} - Brush Size: {brush_size}")
    clock.tick()
    
pygame.quit()

import pygame
import numpy as np
import torch
import torch.nn as nn

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 255, 255
CELL_SIZE = 1
GRID_WIDTH, GRID_HEIGHT = HEIGHT, WIDTH  # Swapped for rotation

# Colors and element types
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

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Falling Sand Game (Model Prediction) - Rotated")

# Create the grid
grid = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=int)

# Define the model architecture
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

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CellularAutomatonCNN().to(device)
model.load_state_dict(torch.load('best_sand_model.pth', map_location=device))
model.eval()

def predict_next_frame(model, current_frame):
    with torch.no_grad():
        current_frame = torch.from_numpy(current_frame).unsqueeze(0).to(device)
        current_frame = torch.nn.functional.one_hot(current_frame.long(), num_classes=7).permute(0, 3, 1, 2).float()
        output = model(current_frame)
        return torch.argmax(output, dim=1).squeeze().cpu().numpy()

def draw_particles():
    surface = pygame.Surface((WIDTH, HEIGHT))
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            color = COLORS[grid[x, y]]
            surface.set_at((GRID_HEIGHT - 1 - y, x), color)  # Rotate 90 degrees counterclockwise
    screen.blit(surface, (0, 0))

running = True
current_element = SAND
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                y, x = pygame.mouse.get_pos()
                y = GRID_HEIGHT - 1 - y  # Rotate 90 degrees counterclockwise
                grid[x, y] = current_element
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:
                y, x = pygame.mouse.get_pos()
                y = GRID_HEIGHT - 1 - y  # Rotate 90 degrees counterclockwise
                grid[x, y] = current_element
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

    # Update grid using the model
    grid = predict_next_frame(model, grid)
    draw_particles()
    pygame.display.flip()
    
    fps = clock.get_fps()
    pygame.display.set_caption(f"FPS: {fps:.2f}")
    clock.tick(30)  # Limit to 30 FPS

pygame.quit()

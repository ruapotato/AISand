import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Constants
WIDTH, HEIGHT = 255, 255
CELL_SIZE = 1
GRID_WIDTH, GRID_HEIGHT = WIDTH, HEIGHT

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

# Diffusion model definition
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
        t = t.unsqueeze(-1).unsqueeze(-1)
        
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

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleDiffusionModel().to(device)
model.load_state_dict(torch.load("./falling_sand_diffusion_model.pth", map_location=device))
model.eval()

# Prediction function
@torch.no_grad()
def predict_next_frame(model, current_frame):
    model.eval()
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

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Diffusion-based Falling Sand Game")

# Create initial grid
grid = torch.zeros((1, 3, HEIGHT, WIDTH), device=device)

# Helper functions
def add_element(x, y, element):
    color = torch.tensor(COLORS[element], device=device).float() / 255.0
    grid[0, :, y, x] = color

def get_element(x, y):
    pixel = grid[0, :, y, x].cpu().numpy()
    distances = [np.linalg.norm(pixel - np.array(color)/255.0) for color in COLORS.values()]
    return np.argmin(distances)

def draw_grid():
    pygame_surface = pygame.surfarray.make_surface((grid[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    screen.blit(pygame_surface, (0, 0))

# Main game loop
running = True
current_element = SAND
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                x, y = pygame.mouse.get_pos()
                add_element(x, y, current_element)
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:
                x, y = pygame.mouse.get_pos()
                add_element(x, y, current_element)
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

    # Generate next frame using the diffusion model
    grid = predict_next_frame(model, grid)

    # Draw the grid
    draw_grid()
    pygame.display.flip()
    
    fps = clock.get_fps()
    pygame.display.set_caption(f"Diffusion-based Falling Sand Game - FPS: {fps:.2f}")
    clock.tick(30)  # Limit to 30 FPS

pygame.quit()

import pygame
import numpy as np
import random
import os
from PIL import Image

# Initialize Pygame
pygame.init()

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

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Falling Sand Game")

# Create the grid
grid = np.zeros((WIDTH, HEIGHT), dtype=int)

def create_random_shape(element, max_size=50):
    shape_type = random.choice(["circle", "rectangle"])
    x = random.randint(0, WIDTH - 1)
    y = random.randint(0, HEIGHT - 1)
    
    if shape_type == "circle":
        radius = random.randint(5, max_size // 2)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                        grid[nx, ny] = element
    else:  # rectangle
        w = random.randint(5, max_size)
        h = random.randint(5, max_size)
        for dx in range(w):
            for dy in range(h):
                nx, ny = x + dx, y + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    grid[nx, ny] = element

def create_random_start():
    global grid
    grid = np.zeros((WIDTH, HEIGHT), dtype=int)
    
    # Create shapes for each element
    for element in [SAND, WATER, PLANT, WOOD, ACID, FIRE]:
        num_shapes = random.randint(1, 5)
        for _ in range(num_shapes):
            create_random_shape(element)

def update_sand(x, y):
    if y < HEIGHT - 1:
        if grid[x, y + 1] == EMPTY:
            grid[x, y + 1] = SAND
            grid[x, y] = EMPTY
            return True
        elif x > 0 and grid[x - 1, y + 1] == EMPTY:
            grid[x - 1, y + 1] = SAND
            grid[x, y] = EMPTY
            return True
        elif x < WIDTH - 1 and grid[x + 1, y + 1] == EMPTY:
            grid[x + 1, y + 1] = SAND
            grid[x, y] = EMPTY
            return True
    return False

def update_water(x, y):
    if y < HEIGHT - 1:
        if grid[x, y + 1] == EMPTY:
            grid[x, y + 1] = WATER
            grid[x, y] = EMPTY
            return True
        else:
            dx = random.choice([-1, 1])
            if 0 <= x + dx < WIDTH and grid[x + dx, y] == EMPTY:
                grid[x + dx, y] = WATER
                grid[x, y] = EMPTY
                return True
    return False

def update_plant(x, y):
    growth_chance = 0.1
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
            if grid[nx, ny] == WATER:
                if random.random() < growth_chance:
                    grid[nx, ny] = PLANT
                    return True
    return False

def update_acid(x, y):
    if y < HEIGHT - 1:
        if grid[x, y + 1] in [EMPTY, PLANT, WOOD]:
            grid[x, y + 1] = ACID
            grid[x, y] = EMPTY
            return True
        else:
            dx = random.choice([-1, 1])
            if 0 <= x + dx < WIDTH and grid[x + dx, y] in [EMPTY, PLANT, WOOD]:
                grid[x + dx, y] = ACID
                grid[x, y] = EMPTY
                return True
    if random.random() < 0.005:
        grid[x, y] = EMPTY
        return True
    return False

def update_fire(x, y):
    moved = False
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                if grid[nx, ny] in [WOOD, PLANT] and random.random() < 0.1:
                    grid[nx, ny] = FIRE
                    moved = True
    if random.random() < 0.05:
        grid[x, y] = EMPTY
        moved = True
    return moved

def update_grid():
    moved = 0
    for x in range(WIDTH):
        for y in range(HEIGHT - 1, -1, -1):
            if grid[x, y] == SAND:
                moved += update_sand(x, y)
            elif grid[x, y] == WATER:
                moved += update_water(x, y)
            elif grid[x, y] == PLANT:
                moved += update_plant(x, y)
            elif grid[x, y] == ACID:
                moved += update_acid(x, y)
            elif grid[x, y] == FIRE:
                moved += update_fire(x, y)
    return moved

def draw_particles():
    surface = pygame.Surface((WIDTH, HEIGHT))
    for x in range(WIDTH):
        for y in range(HEIGHT):
            color = COLORS[grid[x, y]]
            surface.set_at((x, y), color)
    screen.blit(surface, (0, 0))
    return surface

def save_frame(surface, folder, frame_number):
    pygame_image = pygame.surfarray.array3d(surface)
    image = Image.fromarray(pygame_image.transpose(1, 0, 2))
    image.save(f"{folder}/frame_{frame_number:04d}.png")

# Main simulation loop
num_simulations = 100
main_folder = "falling_sand_frames"
os.makedirs(main_folder, exist_ok=True)

MAX_FRAMES = 2000
MOVEMENT_THRESHOLD = 10

for sim in range(num_simulations):
    # Create a subfolder for this simulation
    sim_folder = os.path.join(main_folder, f"simulation_{sim:03d}")
    os.makedirs(sim_folder, exist_ok=True)
    
    create_random_start()
    
    frame_number = 0
    
    while frame_number < MAX_FRAMES:
        surface = draw_particles()
        save_frame(surface, sim_folder, frame_number)
        
        moved = update_grid()
        frame_number += 1
        
        if moved <= MOVEMENT_THRESHOLD:
            print(f"Simulation {sim + 1} settled after {frame_number} frames (low movement)")
            break
        
        pygame.display.flip()
    
    if frame_number == MAX_FRAMES:
        print(f"Simulation {sim + 1} reached maximum frames ({MAX_FRAMES})")
    
    # Save one more frame after settling or reaching max frames
    surface = draw_particles()
    save_frame(surface, sim_folder, frame_number)

pygame.quit()
print("Data collection complete.")

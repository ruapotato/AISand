import pygame
import numpy as np
import random
import os
from PIL import Image
import scipy.signal

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 255, 255
CELL_SIZE = 1
GRID_WIDTH, GRID_HEIGHT = WIDTH, HEIGHT

# Colors and element types
EMPTY, SAND, WATER, PLANT, WOOD, ACID, FIRE, STEAM, SALT, TNT, WAX, OIL, LAVA, STONE = range(14)

# Base colors (R, G, B) - Using distinct colors for each element
BASE_COLORS = {
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
pygame.display.set_caption("Curriculum-based Falling Sand Simulation")

# Create the grids
element_grid = np.zeros((WIDTH, HEIGHT), dtype=int)
temperature_grid = np.zeros((WIDTH, HEIGHT), dtype=float)

def create_square_shape(element, x, y, width, height):
    for dx in range(width):
        for dy in range(height):
            nx, ny = x + dx, y + dy
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                element_grid[nx, ny] = element
                if element in [FIRE, LAVA]:
                    temperature_grid[nx, ny] = random.uniform(800, 1200)
                else:
                    temperature_grid[nx, ny] = 20.0  # Room temperature

def create_circle_shape(element, center_x, center_y, radius):
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                nx, ny = center_x + dx, center_y + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    element_grid[nx, ny] = element
                    if element in [FIRE, LAVA]:
                        temperature_grid[nx, ny] = random.uniform(800, 1200)
                    else:
                        temperature_grid[nx, ny] = 20.0  # Room temperature

def create_single_element_scenario(element):
    global element_grid, temperature_grid
    element_grid = np.zeros((WIDTH, HEIGHT), dtype=int)
    temperature_grid = np.full((WIDTH, HEIGHT), 20.0)
    
    if random.choice([True, False]):  # Randomly choose between square and circle
        shape_width = random.randint(30, 50)
        shape_height = random.randint(30, 50)
        x = random.randint(0, WIDTH - shape_width)
        y = random.randint(0, HEIGHT - shape_height)
        create_square_shape(element, x, y, shape_width, shape_height)
    else:
        radius = random.randint(15, 25)
        center_x = random.randint(radius, WIDTH - radius)
        center_y = random.randint(radius, HEIGHT - radius)
        create_circle_shape(element, center_x, center_y, radius)

def create_two_element_scenario(element1, element2):
    global element_grid, temperature_grid
    element_grid = np.zeros((WIDTH, HEIGHT), dtype=int)
    temperature_grid = np.full((WIDTH, HEIGHT), 20.0)
    
    for element in [element1, element2]:
        if random.choice([True, False]):  # Randomly choose between square and circle
            shape_width = random.randint(30, 50)
            shape_height = random.randint(30, 50)
            x = random.randint(0, WIDTH - shape_width)
            y = random.randint(0, HEIGHT - shape_height)
            create_square_shape(element, x, y, shape_width, shape_height)
        else:
            radius = random.randint(15, 25)
            center_x = random.randint(radius, WIDTH - radius)
            center_y = random.randint(radius, HEIGHT - radius)
            create_circle_shape(element, center_x, center_y, radius)

def create_all_elements_scenario():
    global element_grid, temperature_grid
    element_grid = np.zeros((WIDTH, HEIGHT), dtype=int)
    temperature_grid = np.full((WIDTH, HEIGHT), 20.0)
    
    elements = [SAND, WATER, PLANT, WOOD, ACID, FIRE, STEAM, SALT, TNT, WAX, OIL, LAVA, STONE]
    random.shuffle(elements)
    
    band_height = HEIGHT // len(elements)
    for i, element in enumerate(elements):
        if random.choice([True, False]):  # Randomly choose between square and circle
            create_square_shape(element, 0, i * band_height, WIDTH, band_height)
        else:
            num_circles = random.randint(3, 6)
            for _ in range(num_circles):
                radius = random.randint(10, 20)
                center_x = random.randint(radius, WIDTH - radius)
                center_y = random.randint(i * band_height + radius, (i + 1) * band_height - radius)
                create_circle_shape(element, center_x, center_y, radius)

def update_temperature():
    kernel = np.array([[0.05, 0.1, 0.05],
                       [0.1, 0.4, 0.1],
                       [0.05, 0.1, 0.05]])
    
    global temperature_grid
    temperature_grid = scipy.signal.convolve2d(temperature_grid, kernel, mode='same', boundary='wrap')
    temperature_grid = np.maximum(20, temperature_grid * 0.99)

def get_color(x, y):
    base_color = BASE_COLORS[element_grid[x, y]]
    temp = temperature_grid[x, y]
    alpha = int(np.clip((temp - 20) / (1200 - 20) * 100 + 155, 155, 255))
    return base_color + (alpha,)

def update_movable(x, y, element):
    if y < HEIGHT - 1:
        if element_grid[x, y + 1] == EMPTY:
            element_grid[x, y + 1] = element
            element_grid[x, y] = EMPTY
            temperature_grid[x, y + 1] = temperature_grid[x, y]
            return True
        elif x > 0 and element_grid[x - 1, y + 1] == EMPTY:
            element_grid[x - 1, y + 1] = element
            element_grid[x, y] = EMPTY
            temperature_grid[x - 1, y + 1] = temperature_grid[x, y]
            return True
        elif x < WIDTH - 1 and element_grid[x + 1, y + 1] == EMPTY:
            element_grid[x + 1, y + 1] = element
            element_grid[x, y] = EMPTY
            temperature_grid[x + 1, y + 1] = temperature_grid[x, y]
            return True
    return False

def update_liquid(x, y, element):
    if update_movable(x, y, element):
        return True
    else:
        dx = random.choice([-1, 1])
        if 0 <= x + dx < WIDTH and element_grid[x + dx, y] == EMPTY:
            element_grid[x + dx, y] = element
            element_grid[x, y] = EMPTY
            temperature_grid[x + dx, y] = temperature_grid[x, y]
            return True
    return False

def update_water(x, y):
    if update_liquid(x, y, WATER):
        return True
    if temperature_grid[x, y] >= 100 and random.random() < 0.1:
        element_grid[x, y] = STEAM
        temperature_grid[x, y] -= 50
        return True
    return False

def update_plant(x, y):
    growth_chance = 0.05
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
            if element_grid[nx, ny] == WATER:
                if random.random() < growth_chance:
                    element_grid[nx, ny] = PLANT
                    return True
    if temperature_grid[x, y] > 300 and random.random() < 0.1:
        element_grid[x, y] = FIRE
        temperature_grid[x, y] = random.uniform(800, 1000)
        return True
    return False

def update_fire(x, y):
    moved = False
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                if element_grid[nx, ny] in [WOOD, PLANT, OIL, TNT] and random.random() < 0.1:
                    element_grid[nx, ny] = FIRE
                    temperature_grid[nx, ny] = random.uniform(800, 1000)
                    moved = True
                elif element_grid[nx, ny] == WATER:
                    element_grid[x, y] = STEAM
                    temperature_grid[x, y] = 100
                    element_grid[nx, ny] = EMPTY
                    temperature_grid[nx, ny] = 100
                    return True
    if random.random() < 0.05:
        element_grid[x, y] = EMPTY
        temperature_grid[x, y] = max(100, temperature_grid[x, y] - random.uniform(50, 100))
        moved = True
    return moved

def update_lava(x, y):
    if update_liquid(x, y, LAVA):
        return True
    if temperature_grid[x, y] < 700:
        if random.random() < 0.1:
            element_grid[x, y] = STONE
            return True
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
            if element_grid[nx, ny] in [WOOD, PLANT, OIL]:
                element_grid[nx, ny] = FIRE
                temperature_grid[nx, ny] = random.uniform(800, 1000)
                return True
    return False

def update_grid():
    moved = 0
    for x in range(WIDTH):
        for y in range(HEIGHT - 1, -1, -1):
            if element_grid[x, y] == SAND:
                moved += update_movable(x, y, SAND)
            elif element_grid[x, y] == WATER:
                moved += update_water(x, y)
            elif element_grid[x, y] == PLANT:
                moved += update_plant(x, y)
            elif element_grid[x, y] == FIRE:
                moved += update_fire(x, y)
            elif element_grid[x, y] == LAVA:
                moved += update_lava(x, y)
    return moved

def draw_particles():
    surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for x in range(WIDTH):
        for y in range(HEIGHT):
            color = get_color(x, y)
            surface.set_at((x, y), color)
    screen.blit(surface, (0, 0))
    return surface

def save_frame(surface, folder, frame_number):
    pixel_array = pygame.surfarray.array2d(surface)
    rgba_array = np.zeros((WIDTH, HEIGHT, 4), dtype=np.uint8)
    for x in range(WIDTH):
        for y in range(HEIGHT):
            pixel = pixel_array[x, y]
            r = (pixel >> 16) & 255
            g = (pixel >> 8) & 255
            b = pixel & 255
            a = (pixel >> 24) & 255
            rgba_array[y, x] = [r, g, b, a]
    image = Image.fromarray(rgba_array, 'RGBA')
    image.save(f"{folder}/frame_{frame_number:04d}.png")

def run_simulation(scenario_func, scenario_name, num_simulations):
    main_folder = f"curriculum_falling_sand_frames/{scenario_name}"
    os.makedirs(main_folder, exist_ok=True)

    MAX_FRAMES = 500  # Increased from 100 to 500
    MOVEMENT_THRESHOLD = 30

    for sim in range(num_simulations):
        sim_folder = os.path.join(main_folder, f"simulation_{sim:03d}")
        os.makedirs(sim_folder, exist_ok=True)
        
        scenario_func()
        
        frame_number = 0
        
        while frame_number < MAX_FRAMES:
            surface = draw_particles()
            save_frame(surface, sim_folder, frame_number)
            
            moved = update_grid()
            update_temperature()
            frame_number += 1
            
            if moved <= MOVEMENT_THRESHOLD:
                print(f"{scenario_name} - Simulation {sim + 1} settled after {frame_number} frames (low movement)")
                break
            
            pygame.display.flip()
        
        if frame_number == MAX_FRAMES:
            print(f"{scenario_name} - Simulation {sim + 1} reached maximum frames ({MAX_FRAMES})")
        
        surface = draw_particles()
        save_frame(surface, sim_folder, frame_number)

# Curriculum simulations
elements = [SAND, WATER, PLANT, WOOD, ACID, FIRE, STEAM, SALT, TNT, WAX, OIL, LAVA, STONE]

# Single element simulations
for element in elements:
    run_simulation(lambda: create_single_element_scenario(element), f"single_{BASE_COLORS[element]}", 5)

# Two element simulations
for i, element1 in enumerate(elements):
    for element2 in elements[i+1:]:
        run_simulation(lambda: create_two_element_scenario(element1, element2), f"pair_{BASE_COLORS[element1]}_{BASE_COLORS[element2]}", 5)

# All elements simulation
run_simulation(create_all_elements_scenario, "all_elements", 10)

pygame.quit()
print("Curriculum-based data collection complete.")

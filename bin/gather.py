import pygame
import numpy as np
import random
import os
from PIL import Image
import scipy.signal
import argparse

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

# Global variables for grids
element_grid = np.zeros((WIDTH, HEIGHT), dtype=int)
temperature_grid = np.zeros((WIDTH, HEIGHT), dtype=float)

def create_square_shape(element, x, y, width, height):
    global element_grid, temperature_grid
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
    global element_grid, temperature_grid
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

def create_random_setup(element):
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
    
    # Decide on shape and size for both elements
    shape1 = random.choice(['square', 'circle'])
    shape2 = random.choice(['square', 'circle'])
    
    if shape1 == 'square':
        width1 = random.randint(30, 50)
        height1 = random.randint(30, 50)
    else:
        radius1 = random.randint(15, 25)
        
    if shape2 == 'square':
        width2 = random.randint(30, 50)
        height2 = random.randint(30, 50)
    else:
        radius2 = random.randint(15, 25)
    
    # Decide on placement strategy
    placement = random.choice(['vertical', 'horizontal'])
    
    if placement == 'vertical':
        # Place elements vertically
        if shape1 == 'square':
            x1 = random.randint(0, WIDTH - width1)
            y1 = 0
            create_square_shape(element1, x1, y1, width1, height1)
            
            if shape2 == 'square':
                x2 = max(0, min(x1 + random.randint(-20, 20), WIDTH - width2))
                y2 = HEIGHT - height2
            else:
                x2 = max(radius2, min(x1 + random.randint(-20, 20), WIDTH - radius2))
                y2 = HEIGHT - radius2
        else:
            x1 = random.randint(radius1, WIDTH - radius1)
            y1 = radius1
            create_circle_shape(element1, x1, y1, radius1)
            
            if shape2 == 'square':
                x2 = max(0, min(x1 + random.randint(-20, 20), WIDTH - width2))
                y2 = HEIGHT - height2
            else:
                x2 = max(radius2, min(x1 + random.randint(-20, 20), WIDTH - radius2))
                y2 = HEIGHT - radius2
    else:
        # Place elements horizontally
        if shape1 == 'square':
            x1 = 0
            y1 = random.randint(0, HEIGHT - height1)
            create_square_shape(element1, x1, y1, width1, height1)
            
            if shape2 == 'square':
                x2 = WIDTH - width2
                y2 = max(0, min(y1 + random.randint(-20, 20), HEIGHT - height2))
            else:
                x2 = WIDTH - radius2
                y2 = max(radius2, min(y1 + random.randint(-20, 20), HEIGHT - radius2))
        else:
            x1 = radius1
            y1 = random.randint(radius1, HEIGHT - radius1)
            create_circle_shape(element1, x1, y1, radius1)
            
            if shape2 == 'square':
                x2 = WIDTH - width2
                y2 = max(0, min(y1 + random.randint(-20, 20), HEIGHT - height2))
            else:
                x2 = WIDTH - radius2
                y2 = max(radius2, min(y1 + random.randint(-20, 20), HEIGHT - radius2))
    
    # Create the second shape
    if shape2 == 'square':
        create_square_shape(element2, x2, y2, width2, height2)
    else:
        create_circle_shape(element2, x2, y2, radius2)

    # Ensure elements are touching by filling the gap if necessary
    if placement == 'vertical':
        min_y1 = y1 if shape1 == 'square' else y1 - radius1
        max_y1 = y1 + height1 if shape1 == 'square' else y1 + radius1
        min_y2 = y2 if shape2 == 'square' else y2 - radius2
        for y in range(max_y1, min_y2):
            element_grid[x1, y] = element1
            temperature_grid[x1, y] = 20.0 if element1 not in [FIRE, LAVA] else random.uniform(800, 1200)
    else:
        min_x1 = x1 if shape1 == 'square' else x1 - radius1
        max_x1 = x1 + width1 if shape1 == 'square' else x1 + radius1
        min_x2 = x2 if shape2 == 'square' else x2 - radius2
        for x in range(max_x1, min_x2):
            element_grid[x, y1] = element1
            temperature_grid[x, y1] = 20.0 if element1 not in [FIRE, LAVA] else random.uniform(800, 1200)

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
                radius = min(random.randint(5, 10), band_height // 2 - 1)  # Ensure radius fits within band
                center_x = random.randint(radius, WIDTH - radius)
                min_y = max(i * band_height + radius, 0)
                max_y = min((i + 1) * band_height - radius, HEIGHT - 1)
                
                if min_y < max_y:
                    center_y = random.randint(min_y, max_y)
                    create_circle_shape(element, center_x, center_y, radius)
                else:
                    # Fallback to creating a small rectangle if circle doesn't fit
                    rect_height = min(band_height, 5)
                    rect_y = i * band_height + (band_height - rect_height) // 2
                    create_square_shape(element, center_x - radius, rect_y, radius * 2, rect_height)

def update_temperature():
    global temperature_grid
    kernel = np.array([[0.05, 0.1, 0.05],
                       [0.1, 0.4, 0.1],
                       [0.05, 0.1, 0.05]])
    
    temperature_grid = scipy.signal.convolve2d(temperature_grid, kernel, mode='same', boundary='wrap')
    temperature_grid = np.maximum(20, temperature_grid * 0.99)

def get_color(x, y):
    base_color = BASE_COLORS[element_grid[x, y]]
    temp = temperature_grid[x, y]
    alpha = int(np.clip((temp - 20) / (1200 - 20) * 100 + 155, 155, 255))
    return base_color + (alpha,)

def update_movable(x, y, element):
    global element_grid, temperature_grid
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
    global element_grid, temperature_grid
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
    global element_grid, temperature_grid
    if update_liquid(x, y, WATER):
        return True
    if temperature_grid[x, y] >= 100 and random.random() < 0.1:
        element_grid[x, y] = STEAM
        temperature_grid[x, y] -= 50
        return True
    return False

def update_plant(x, y):
    global element_grid, temperature_grid
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
    global element_grid, temperature_grid
    moved = False
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                if element_grid[nx, ny] in [WOOD, PLANT, OIL] and random.random() < 0.1:
                    element_grid[nx, ny] = FIRE
                    temperature_grid[nx, ny] = random.uniform(800, 1000)
                    moved = True
                elif element_grid[nx, ny] == WATER:
                    element_grid[x, y] = STEAM
                    temperature_grid[x, y] = 100
                    element_grid[nx, ny] = EMPTY
                    temperature_grid[nx, ny] = 100
                    return True
                elif element_grid[nx, ny] == TNT and random.random() < 0.05:
                    explode(nx, ny)
                    return True
    if random.random() < 0.05:
        element_grid[x, y] = EMPTY
        temperature_grid[x, y] = max(100, temperature_grid[x, y] - random.uniform(50, 100))
        moved = True
    return moved

def update_lava(x, y):
    global element_grid, temperature_grid
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
            elif element_grid[nx, ny] == WATER:
                element_grid[nx, ny] = STEAM
                temperature_grid[nx, ny] = 100
                element_grid[x, y] = STONE
                temperature_grid[x, y] = 700
                return True
    return False

def update_acid(x, y):
    global element_grid, temperature_grid
    if update_liquid(x, y, ACID):
        return True
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
            if element_grid[nx, ny] in [WOOD, PLANT, STONE] and random.random() < 0.1:
                element_grid[nx, ny] = EMPTY
                element_grid[x, y] = EMPTY
                return True
    return False

def update_steam(x, y):
    global element_grid, temperature_grid
    if x > 0 and element_grid[x - 1, y] == EMPTY:
        element_grid[x - 1, y] = STEAM
        element_grid[x, y] = EMPTY
        temperature_grid[x - 1, y] = temperature_grid[x, y]
        return True
    elif y > 0 and element_grid[x, y - 1] == EMPTY:
        element_grid[x, y - 1] = STEAM
        element_grid[x, y] = EMPTY
        temperature_grid[x, y - 1] = temperature_grid[x, y]
        return True
    elif y < HEIGHT - 1 and element_grid[x, y + 1] == EMPTY:
        element_grid[x, y + 1] = STEAM
        element_grid[x, y] = EMPTY
        temperature_grid[x, y + 1] = temperature_grid[x, y]
        return True
    if temperature_grid[x, y] < 100 and random.random() < 0.1:
        element_grid[x, y] = WATER
        return True
    return False

def explode(x, y):
    global element_grid, temperature_grid
    radius = 10
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                nx, ny = x + dx, y + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    if random.random() < 0.7:
                        element_grid[nx, ny] = FIRE
                        temperature_grid[nx, ny] = random.uniform(800, 1200)
                    else:
                        element_grid[nx, ny] = EMPTY
                        temperature_grid[nx, ny] = random.uniform(400, 600)

def update_grid():
    moved = 0
    # Update steam from right to left
    for x in range(WIDTH - 1, -1, -1):
        for y in range(HEIGHT):
            if element_grid[x, y] == STEAM:
                moved += update_steam(x, y)
    
    # Update other elements
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
            elif element_grid[x, y] == ACID:
                moved += update_acid(x, y)
            elif element_grid[x, y] == OIL:
                moved += update_liquid(x, y, OIL)
    return moved


def draw_particles(screen):
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

def run_simulation(scenario_func, scenario_name, num_images, process_id, output_dir):
    # Set up the display for this process
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Falling Sand Simulation - Process {process_id}")

    if output_dir:
        main_folder = f"curriculum_falling_sand_frames/{output_dir}/{scenario_name}"
    else:
        main_folder = f"curriculum_falling_sand_frames/{scenario_name}"
    os.makedirs(main_folder, exist_ok=True)

    MOVEMENT_THRESHOLD = 30
    MIN_FRAMES = 3
    MAX_FRAMES = 100

    frame_number = 0
    total_scenarios = 0

    while frame_number < num_images:
        scenario_func()
        total_scenarios += 1
        
        scenario_frames = 0
        scenario_movement = 0
        
        while scenario_frames < MAX_FRAMES and frame_number < num_images:
            surface = draw_particles(screen)
            
            sim_folder = os.path.join(main_folder, f"simulation_{total_scenarios:03d}")
            os.makedirs(sim_folder, exist_ok=True)
            save_frame(surface, sim_folder, scenario_frames)
            
            moved = update_grid()
            update_temperature()
            frame_number += 1
            scenario_frames += 1
            scenario_movement += moved
            
            pygame.display.flip()

            if scenario_frames >= MIN_FRAMES and moved <= MOVEMENT_THRESHOLD:
                break

        print(f"{scenario_name} - Process {process_id} - Scenario {total_scenarios}: {scenario_frames} frames, {scenario_movement} total movement")

        if frame_number >= num_images:
            break

    print(f"{scenario_name} - Process {process_id} - Simulation complete. {frame_number} total frames, {total_scenarios} scenarios")

def main(args):
    total_images_per_element = args.images
    elements = [SAND, WATER, PLANT, WOOD, ACID, FIRE, STEAM, SALT, TNT, WAX, OIL, LAVA, STONE]

    if args.segment == 'single':
        for element in elements[args.start:args.end]:
            scenario_func = lambda: create_random_setup(element)
            scenario_name = f"single_{BASE_COLORS[element]}"
            run_simulation(scenario_func, scenario_name, total_images_per_element, args.process_id, args.output_dir)

    elif args.segment == 'pair':
        for i, element1 in enumerate(elements[args.start:args.end]):
            for element2 in elements[i+1:]:
                scenario_func = lambda: create_two_element_scenario(element1, element2)
                scenario_name = f"pair_{BASE_COLORS[element1]}_{BASE_COLORS[element2]}"
                run_simulation(scenario_func, scenario_name, total_images_per_element, args.process_id, args.output_dir)

    elif args.segment == 'all':
        scenario_name = "all_elements"
        run_simulation(create_all_elements_scenario, scenario_name, total_images_per_element, args.process_id, args.output_dir)

    print(f"Simulation segment {args.segment} (Process {args.process_id}) complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Falling Sand Simulation")
    parser.add_argument('--segment', choices=['single', 'pair', 'all'], required=True, help="Simulation segment to run")
    parser.add_argument('--start', type=int, default=0, help="Start index for elements")
    parser.add_argument('--end', type=int, default=13, help="End index for elements")
    parser.add_argument('--images', type=int, default=1000, help="Number of images to generate")
    parser.add_argument('--process_id', type=int, required=True, help="Process ID for this instance")
    parser.add_argument('--output_dir', type=str, default='', help="Output directory for the simulation")
    args = parser.parse_args()

    main(args)

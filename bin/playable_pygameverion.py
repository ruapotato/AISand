import pygame
import numpy as np
import random

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

def update_sand(x, y):
    if y < HEIGHT - 1:
        if grid[x, y + 1] == EMPTY:
            grid[x, y + 1] = SAND
            grid[x, y] = EMPTY
        elif x > 0 and grid[x - 1, y + 1] == EMPTY:
            grid[x - 1, y + 1] = SAND
            grid[x, y] = EMPTY
        elif x < WIDTH - 1 and grid[x + 1, y + 1] == EMPTY:
            grid[x + 1, y + 1] = SAND
            grid[x, y] = EMPTY

def update_water(x, y):
    if y < HEIGHT - 1:
        if grid[x, y + 1] == EMPTY:
            grid[x, y + 1] = WATER
            grid[x, y] = EMPTY
        else:
            dx = random.choice([-1, 1])
            if 0 <= x + dx < WIDTH and grid[x + dx, y] == EMPTY:
                grid[x + dx, y] = WATER
                grid[x, y] = EMPTY

def update_plant(x, y):
    growth_chance = 0.1  # Increased growth chance
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
            if grid[nx, ny] == WATER:
                if random.random() < growth_chance:
                    grid[nx, ny] = PLANT
                    break

def update_acid(x, y):
    if y < HEIGHT - 1:
        if grid[x, y + 1] in [EMPTY, PLANT, WOOD]:
            grid[x, y + 1] = ACID
            grid[x, y] = EMPTY
        else:
            dx = random.choice([-1, 1])
            if 0 <= x + dx < WIDTH and grid[x + dx, y] in [EMPTY, PLANT, WOOD]:
                grid[x + dx, y] = ACID
                grid[x, y] = EMPTY
    if random.random() < 0.005:
        grid[x, y] = EMPTY

def update_fire(x, y):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                if grid[nx, ny] in [WOOD, PLANT] and random.random() < 0.1:
                    grid[nx, ny] = FIRE
    if random.random() < 0.05:
        grid[x, y] = EMPTY

def update_grid():
    for x in range(WIDTH):
        for y in range(HEIGHT - 1, -1, -1):
            if grid[x, y] == SAND:
                update_sand(x, y)
            elif grid[x, y] == WATER:
                update_water(x, y)
            elif grid[x, y] == PLANT:
                update_plant(x, y)
            elif grid[x, y] == ACID:
                update_acid(x, y)
            elif grid[x, y] == FIRE:
                update_fire(x, y)

def draw_particles():
    surface = pygame.Surface((WIDTH, HEIGHT))
    for x in range(WIDTH):
        for y in range(HEIGHT):
            color = COLORS[grid[x, y]]
            surface.set_at((x, y), color)
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
                x, y = pygame.mouse.get_pos()
                grid[x, y] = current_element
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:
                x, y = pygame.mouse.get_pos()
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

    update_grid()
    draw_particles()
    pygame.display.flip()
    
    fps = clock.get_fps()
    pygame.display.set_caption(f"Falling Sand Game - FPS: {fps:.2f}")
    clock.tick()

pygame.quit()

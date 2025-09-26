import pygame
import numpy as np
import math

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 1000, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hexagon Tiling")
clock = pygame.time.Clock()

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
HEX_COLOR = (200, 200, 255)

# --- Hexagon Properties ---
R = 30  # Radius (distance from center to corner)
W = math.sqrt(3) * R  # Width of hexagon
H = 2 * R             # Height of hexagon
VERTICAL_SPACING = 0.75 * H  # vertical spacing between rows

# --- Function to compute hexagon vertices ---
def hexagon_points(cx, cy, R):
    points = []
    for i in range(6):
        angle_deg = 60 * i - 30  # flat-topped orientation
        angle_rad = math.radians(angle_deg)
        x = cx + R * math.cos(angle_rad)
        y = cy + R * math.sin(angle_rad)
        points.append((x, y))
    return points

# --- Main Loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    # --- Draw Tiled Hexagons ---
    row = 0
    y = 0
    while y < HEIGHT + H:
        x_offset = (W / 2) if (row % 2 == 1) else 0  # stagger every other row
        x = x_offset
        while x < WIDTH + W:
            points = hexagon_points(x, y, R)
            pygame.draw.polygon(screen, HEX_COLOR, points)
            pygame.draw.polygon(screen, WHITE, points, 2)
            x += W
        y += VERTICAL_SPACING
        row += 1

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

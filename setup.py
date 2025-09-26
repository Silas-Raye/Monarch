import pygame
import numpy as np
import math
import os

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 1080, 680
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Isometric Hexagonal Prism - Tiled")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 20)

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# --- Face Colors ---
SIDE_FACE_COLOR = (200, 200, 255)
TOP_FACE_COLOR = (120, 180, 120)

# --- Isometric Projection Matrix ---
iso_angle = math.radians(30)
iso_matrix = np.array([
    [math.cos(iso_angle), -math.cos(iso_angle), 0],
    [math.sin(iso_angle), math.sin(iso_angle), -1]
])

# Parameters for draw tiled prisms
cell_radius = 30
cell_height = 30
num_cols = 15
num_rows = 15
face_dir = os.path.join(os.path.dirname(__file__), "exported_faces")
image_paths = {
    "top": os.path.join(face_dir, "face_top.png"),
    0: os.path.join(face_dir, "face_0.png"),
    1: os.path.join(face_dir, "face_1.png"),
    5: os.path.join(face_dir, "face_5.png"),
}
plant_paths = {
    0: os.path.join(face_dir, "face_0.png"),
}

# Export a dictionary with all the necessary setup variables
game_setup = {
    "screen": screen,
    "clock": clock,
    "font": font,
    "WIDTH": WIDTH,
    "HEIGHT": HEIGHT,
    "CENTER_X": WIDTH // 2,
    "CENTER_Y": HEIGHT // 2,
    "BLACK": BLACK,
    "WHITE": WHITE,
    "SIDE_FACE_COLOR": SIDE_FACE_COLOR,
    "TOP_FACE_COLOR": TOP_FACE_COLOR,
    "iso_matrix": iso_matrix,
    "cell_radius": cell_radius,
    "cell_height": cell_height,
    "num_cols": num_cols,
    "num_rows": num_rows,
    "image_paths": image_paths,
    "plant_paths": plant_paths
}

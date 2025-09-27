import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np
import math
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current file
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to the parent directory
sys.path.append(parent_dir) # Add the parent directory to the Python path

# Now you can import 'setup' and 'functions' from the parent directory
from setup import game_setup
from functions import project_3d_to_2d, get_face_color, save_visible_faces

# Access the setup variables from the imported dictionary
screen = game_setup["screen"]
clock = game_setup["clock"]
font = game_setup["font"]
WIDTH = game_setup["WIDTH"]
HEIGHT = game_setup["HEIGHT"]
CENTER_X = game_setup["CENTER_X"]
CENTER_Y = game_setup["CENTER_Y"]
BLACK = game_setup["BLACK"]
WHITE = game_setup["WHITE"]
iso_matrix = game_setup["iso_matrix"]
cell_radius = game_setup["cell_radius"]
cell_height = game_setup["cell_height"]

# --- Define Vertices (3D coordinates) ---
vertices = []
for i in range(6):
    angle_deg = 60 * i
    angle_rad = math.radians(angle_deg)
    x = cell_radius * math.cos(angle_rad)
    y = cell_radius * math.sin(angle_rad)
    # Top face (z > 0)
    vertices.append([x, y, cell_height / 2])
    # Bottom face (z < 0)
    vertices.append([x, y, -cell_height / 2])
vertices = np.array(vertices)

# --- Define Faces ---
faces = {
    "top": [0, 2, 4, 6, 8, 10],
    "bottom": [1, 3, 5, 7, 9, 11],
    # Six side rectangles
    0: [0, 1, 3, 2],    # side 0
    1: [2, 3, 5, 4],    # side 1
    2: [4, 5, 7, 6],    # side 2
    3: [6, 7, 9, 8],    # side 3
    4: [8, 9, 11, 10],  # side 4
    5: [10, 11, 1, 0]   # side 5
}

# --- Choose which faces to show ---
visible_faces = ["top", 0, 1, 5]
show_face_indices = True

# --- Game Loop ---
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)
    
    # --- Compute Z for Sorting ---
    face_z_avg = []
    face_keys = []
    for key in visible_faces:
        face_vertices = vertices[faces[key]]
        avg_z = np.mean(face_vertices[:, 2])
        face_z_avg.append(avg_z)
        face_keys.append(key)
    
    sorted_indices = np.argsort(face_z_avg)
    
    # --- Draw Faces ---
    for i in sorted_indices:
        key = face_keys[i]
        face_indices = faces[key]
        projected_face = [project_3d_to_2d(vertices[v_idx]) for v_idx in face_indices]
        
        face_color = get_face_color(key)
        
        pygame.draw.polygon(screen, face_color, projected_face)
        pygame.draw.polygon(screen, WHITE, projected_face, 2)
        
        # Label side faces with their index
        if isinstance(key, int):
            cx = sum(p[0] for p in projected_face) / len(projected_face)
            cy = sum(p[1] for p in projected_face) / len(projected_face)
            # Write the face index in white
            if show_face_indices:
                label = font.render(str(key), True, WHITE)
                screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

# On exit, export faces
save_visible_faces(visible_faces, vertices, faces, project_3d_to_2d)

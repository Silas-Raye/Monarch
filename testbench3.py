import pygame
import numpy as np
import math

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 1000, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Isometric Hexagonal Prism")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)

# --- Prism Properties ---
R = 30  # Radius of the hexagon, corresponds to a 30mm side length
H = 60  # Height of the prism, 60mm
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2

# --- Define Vertices (3D coordinates) ---
vertices = []
for i in range(6):
    angle_deg = 60 * i
    angle_rad = math.radians(angle_deg)
    x = R * math.cos(angle_rad)
    y = R * math.sin(angle_rad)
    # Top face (z > 0)
    vertices.append([x, y, H / 2])
    # Bottom face (z < 0)
    vertices.append([x, y, -H / 2])
vertices = np.array(vertices)

# --- Define Faces ---
faces = {
    "top": [0, 2, 4, 6, 8, 10],
    "bottom": [1, 3, 5, 7, 9, 11],
    # Six side rectangles
    0: [0, 1, 3, 2],     # side 0
    1: [2, 3, 5, 4],     # side 1
    2: [4, 5, 7, 6],     # side 2
    3: [6, 7, 9, 8],     # side 3
    4: [8, 9, 11, 10],   # side 4
    5: [10, 11, 1, 0]    # side 5
}

# --- Choose which faces to show ---
visible_faces = ["top", 0, 1, 5]  # change this list

# --- Isometric Projection Matrix ---
iso_angle = math.radians(30)
iso_matrix = np.array([
    [math.cos(iso_angle), -math.cos(iso_angle), 0],
    [math.sin(iso_angle), math.sin(iso_angle), -1]
])

def project_3d_to_2d(vertex):
    projected_2d = np.dot(iso_matrix, vertex)
    return [
        CENTER_X + projected_2d[0],
        CENTER_Y + projected_2d[1]
    ]

# --- Main Loop ---
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
        
        # Pick color
        if key == "top":
            face_color = (200, 200, 255)
        elif key == "bottom":
            face_color = (80, 80, 80)
        else:
            face_color = (120, 180, 120)
        
        pygame.draw.polygon(screen, face_color, projected_face)
        pygame.draw.polygon(screen, WHITE, projected_face, 2)
        
        # Label side faces with their index
        if isinstance(key, int):
            cx = sum(p[0] for p in projected_face) / len(projected_face)
            cy = sum(p[1] for p in projected_face) / len(projected_face)
            label = font.render(str(key), True, WHITE)
            screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

# the above python code draws a solid 3d hexagonal prism in 2d space from an isometric perspective
# please tile the whole screen with hexagonal prisms like the one above. make sure to draw the tiles in the correct order so that they overlap properly.

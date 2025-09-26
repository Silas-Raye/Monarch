import pygame
import numpy as np
import math
import os
import csv

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 1000, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Isometric Hexagonal Prism - Tiled")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 20)

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# --- Isometric Projection Matrix ---
iso_angle = math.radians(30)
iso_matrix = np.array([
    [math.cos(iso_angle), -math.cos(iso_angle), 0],
    [math.sin(iso_angle), math.sin(iso_angle), -1]
])

CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2

# --- Function Definitions ---

def project_3d_to_2d(vertex):
    """Project a 3D vertex (x,y,z) to 2D isometric screen coordinates."""
    projected_2d = np.dot(iso_matrix, vertex)
    return [CENTER_X + projected_2d[0], CENTER_Y + projected_2d[1]]

def prism_faces(center, R, H):
    """Create the 3D vertices and faces for a hexagonal prism centered at `center`.

    center: (x, y, z)
    R: radius (distance from center to each hex vertex)
    H: height of prism

    Returns: list of (key, vertices3d) where key is "top", "bottom", or side index 0..5
    """
    cx, cy, cz = center
    verts = []
    for i in range(6):
        angle_rad = math.radians(60 * i)
        x = cx + R * math.cos(angle_rad)
        y = cy + R * math.sin(angle_rad)
        verts.append([x, y, cz + H / 2])  # top
        verts.append([x, y, cz - H / 2])  # bottom
    verts = np.array(verts)

    faces = []
    faces.append(("top", verts[[0, 2, 4, 6, 8, 10]]))
    faces.append(("bottom", verts[[1, 3, 5, 7, 9, 11]]))
    # sides
    side_defs = [
        [0, 1, 3, 2],
        [2, 3, 5, 4],
        [4, 5, 7, 6],
        [6, 7, 9, 8],
        [8, 9, 11, 10],
        [10, 11, 1, 0]
    ]
    for idx, s in enumerate(side_defs):
        faces.append((idx, verts[s]))

    return faces

def load_face_images(face_image_paths):
    """Load images from a mapping of key->path or key->Surface.

    Returns dict key->pygame.Surface (with alpha).
    """
    if not face_image_paths:
        return {}
    images = {}
    for k, v in face_image_paths.items():
        try:
            if isinstance(v, str):
                path = v
                if not os.path.isabs(path):
                    # assume relative to this file
                    path = os.path.join(os.path.dirname(__file__), path)
                img = pygame.image.load(path).convert_alpha()
            elif isinstance(v, pygame.Surface):
                img = v
            else:
                continue
            images[k] = img
        except Exception:
            # silently skip missing/invalid files
            continue
    return images

def draw_tiled_prisms(screen, R=None, H=None, cols=None, rows=None, margin=6, face_image_paths=None, map_dir = "exported_maps", viz_map=None, h_map=None, labs=False):
    """Tile the screen with hexagonal prisms and draw them in hex-diagonal order.

    New rendering order (tile traversal):
      - Hex "diagonals" (bands) from top-left outward, at 90° to side 5.
      - Within each diagonal, left-to-right by column.

    Faces within each prism remain: bottom, 2, 3, 4, 1, 0, 5, top.
    """

    # Ensure map directory exists
    os.makedirs(map_dir, exist_ok=True)
    viz_map_path = os.path.join(map_dir, viz_map) if viz_map else None
    h_map_path = os.path.join(map_dir, h_map) if h_map else None

    # ---- Helpers for odd-q (flat-top) offset -> axial mapping ----
    # Your layout uses y = r*dy + (c % 2)*(dy/2), i.e. odd columns are shifted,
    # which is the "odd-q" layout (see Red Blob Games).
    def offset_oddq_to_axial(r, c):
        aq = c
        ar = r - ((c - (c & 1)) // 2)
        return aq, ar

    def diagonal_band_index(r, c):
        aq, ar = offset_oddq_to_axial(r, c)
        # Lines at 90° to face 5 correspond to constant (aq + ar) in axial space.
        return aq + ar

    # ---- Visible faces selection (same as your original) ----
    if face_image_paths:
        visible_faces = set(face_image_paths.keys())
    else:
        visible_faces = set(["top", 0, 1, 5])

    # Flat-top hex spacing
    dx = 1.5 * R
    dy = math.sqrt(3) * R

    # Determine grid size
    if cols is None:
        cols = int(WIDTH / dx) + margin
    if rows is None:
        rows = int(HEIGHT / dy) + margin

    # Center the grid roughly on screen
    grid_offset_x = -cols * dx / 2
    grid_offset_y = -rows * dy / 2

    # Preload any face images once
    face_images = load_face_images(face_image_paths)

    # Optional visibility map
    viz_map = None
    if viz_map_path:
        try:
            viz_map = []
            with open(viz_map_path, newline='') as vf:
                reader = csv.reader(vf)
                for row in reader:
                    if not row:
                        continue
                    viz_map.append([int(x) for x in row])
            if not viz_map:
                viz_map = None
        except Exception:
            viz_map = None

    # Optional height map
    h_map = None
    if h_map_path:
        try:
            h_map = []
            with open(h_map_path, newline='') as hf:
                reader = csv.reader(hf)
                for row in reader:
                    if not row:
                        continue
                    h_map.append([int(x) for x in row])
            if not h_map:
                h_map = None
        except Exception:
            h_map = None

    # Fixed per-prism face draw order
    local_face_order = ["bottom", 2, 3, 4, 1, 0, 5, "top"]

    # ---- Build hex-diagonal traversal order ----
    # Collect all cells with their band index; sort by band, then by column (c), then by row (r)
    # This yields the diagonal-length pattern you described: 1,3,5,..., n, ... ,3,1 (for square grids).
    cells = []
    for r in range(rows):
        for c in range(cols):
            if viz_map is not None:
                try:
                    if viz_map[r][c] == 0:
                        continue
                except Exception:
                    pass
            b = diagonal_band_index(r, c)
            cells.append((b, c, r))  # sort keys: band, then left->right within the band, then row as tiebreaker
    cells.sort(key=lambda t: (t[0], t[1], t[2]))

    # ---- Draw prisms in the new order ----
    for _, c, r in cells:
        # Flat-top layout: every other column is vertically offset
        x = c * dx + grid_offset_x
        y = r * dy + (c % 2) * (dy / 2) + grid_offset_y
        z = 0

        # Per-tile vertical pixel offset
        try:
            h_offset = int(h_map[r][c]) if h_map is not None else 0
        except Exception:
            h_offset = 0

        center = (x, y, z)

        # Get all faces for this prism and index them
        faces_list = prism_faces(center, R, H)
        faces_dict = {key: verts for key, verts in faces_list}

        # Draw faces in fixed order for THIS prism only
        for key in local_face_order:
            if key not in faces_dict:
                continue
            if key not in visible_faces:
                continue

            verts3d = faces_dict[key]

            # Project 3D verts to 2D and apply per-tile vertical pixel offset
            projected = []
            for v in verts3d:
                px, py = project_3d_to_2d(v)
                projected.append((px, py - h_offset))

            # If image exists for this face, texture-map clipped to polygon
            img = face_images.get(key)
            if img is not None:
                xs = [p[0] for p in projected]
                ys = [p[1] for p in projected]
                minx = int(math.floor(min(xs)))
                miny = int(math.floor(min(ys)))
                maxx = int(math.ceil(max(xs)))
                maxy = int(math.ceil(max(ys)))
                pad = 6
                minx -= pad
                miny -= pad
                maxx += pad
                maxy += pad
                w = max(1, maxx - minx)
                h = max(1, maxy - miny)

                try:
                    scaled = pygame.transform.smoothscale(img, (w, h))
                except Exception:
                    scaled = pygame.transform.scale(img, (w, h))

                image_surf = pygame.Surface((w, h), pygame.SRCALPHA)
                image_surf.blit(scaled, (0, 0))

                # create mask polygon (white inside poly, transparent outside)
                mask_surf = pygame.Surface((w, h), pygame.SRCALPHA)
                poly_rel = [(int(x - minx), int(y - miny)) for x, y in projected]
                pygame.draw.polygon(mask_surf, (255, 255, 255, 255), poly_rel)

                # multiply alpha by mask
                image_surf.blit(mask_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

                # blit the textured polygon at the correct position
                screen.blit(image_surf, (minx, miny))
                # outline
                pygame.draw.polygon(screen, WHITE, projected, 1)
            else:
                # Fallback color per face type
                if key == "top":
                    color = (200, 200, 255)
                elif key == "bottom":
                    color = (80, 80, 80)
                else:
                    base = 150
                    shade = int((key % 3) * 20)
                    color = (base - shade, 170 - shade, 130 - shade)

                pygame.draw.polygon(screen, color, projected)
                pygame.draw.polygon(screen, WHITE, projected, 1)

            if labs:
                # If this is the top face, label it with "row,col" (1-based)
                if key == "top":
                    cx = sum(p[0] for p in projected) / len(projected)
                    cy = sum(p[1] for p in projected) / len(projected)

                    label_txt = f"{r+1},{c+1}"
                    label_surf = font.render(label_txt, True, (255, 255, 255))
                    shadow = font.render(label_txt, True, (0, 0, 0))
                    lx = int(cx - label_surf.get_width() // 2)
                    ly = int(cy - label_surf.get_height() // 2) - 2
                    screen.blit(shadow, (lx + 1, ly + 1))
                    screen.blit(label_surf, (lx, ly))

def main():
    running = True
    R = 30
    H = 30
    # visible faces are now inferred from provided face images per function call

    # Example: explicitly set number of columns and rows in the grid.
    # Change these two values to adjust the number of tiles drawn.
    example_cols = 12
    example_rows = 12

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        # Build mapping for exported face images (paths relative to repository)
        face_dir = os.path.join(os.path.dirname(__file__), "exported_faces")
        image_paths = {
            "top": os.path.join(face_dir, "onexit_top.png"),
            0: os.path.join(face_dir, "onexit_0.png"),
            1: os.path.join(face_dir, "onexit_1.png"),
            5: os.path.join(face_dir, "onexit_5.png"),
        }
        
        draw_tiled_prisms(screen, R=R, H=H, cols=example_cols, rows=example_rows, face_image_paths=image_paths, viz_map="viz_map_unif.csv", h_map="h_map_rand.csv", labs=False)
        draw_tiled_prisms(screen, R=R, H=H, cols=example_cols, rows=example_rows, face_image_paths=image_paths, viz_map="viz_map_cust.csv", h_map="h_map_sum.csv", labs=True)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

import pygame
import numpy as np
import math
import os

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


def draw_prism(screen, center, R, H, visible_faces=None, face_image_paths=None):
    """Draw a single hexagonal prism at `center`.

    This function renders faces for one prism only. Faces are depth-sorted by their
    average Z so the prism looks correct in isolation.
    """
    if visible_faces is None:
        visible_faces = ["top", 0, 1, 5]

    # preload images if provided
    face_images = load_face_images(face_image_paths)

    # face_image_paths can be passed in newer signature; support either
    # (backwards-compatible): if caller passed a dict of paths as 5th arg
    # this will be handled by wrapper; keep this function able to accept
    # images via keyword arg by checking local scope in callers.

    faces = prism_faces(center, R, H)
    face_list = []
    for key, verts3d in faces:
        if key not in visible_faces:
            continue
        avg_z = float(np.mean(verts3d[:, 2]))
        face_list.append((avg_z, key, verts3d))

    face_list.sort(key=lambda x: (x[0]))
    for _, key, verts3d in face_list:
        projected = [project_3d_to_2d(v) for v in verts3d]

        img = face_images.get(key)
        if img is not None:
            xs = [p[0] for p in projected]
            ys = [p[1] for p in projected]
            minx = int(math.floor(min(xs)))
            miny = int(math.floor(min(ys)))
            maxx = int(math.ceil(max(xs)))
            maxy = int(math.ceil(max(ys)))
            # add small padding to avoid seams between adjacent textured faces
            pad = 2
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

            mask_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            poly_rel = [(int(x - minx), int(y - miny)) for x, y in projected]
            pygame.draw.polygon(mask_surf, (255, 255, 255, 255), poly_rel)

            image_surf.blit(mask_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            screen.blit(image_surf, (minx, miny))
            pygame.draw.polygon(screen, WHITE, projected, 1)
        else:
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


def draw_tiled_prisms(screen, R=30, H=60, visible_faces=None, cols=None, rows=None, margin=6, face_image_paths=None):
    """Tile the screen with hexagonal prisms and draw them with correct overlap.

    visible_faces: list of face keys to render per prism (e.g. ["top", 0, 1, 5])
    """
    if visible_faces is None:
        visible_faces = ["top", 0, 1, 5]

    # Use flat-top hex spacing (vertex generation starts at angle 0 -> flat-top)
    # For flat-top hexes:
    #  - horizontal spacing between columns = 1.5 * R
    #  - vertical spacing between rows = sqrt(3) * R
    dx = 1.5 * R
    dy = math.sqrt(3) * R

    # Determine how many rows/cols to cover the screen (add margin)
    # You can override the grid size by passing `cols` and/or `rows` to this function.
    if cols is None:
        cols = int(WIDTH / dx) + margin
    if rows is None:
        rows = int(HEIGHT / dy) + margin

    # Center the grid roughly on screen
    grid_offset_x = -cols * dx / 2
    grid_offset_y = -rows * dy / 2

    # Collect all faces from all prisms for global depth sorting
    all_faces = []

    # Preload any face images once for the entire tiled draw
    face_images = load_face_images(face_image_paths)

    for r in range(rows):
        for c in range(cols):
            # For flat-top layout, offset every other column vertically
            x = c * dx + grid_offset_x
            y = r * dy + (c % 2) * (dy / 2) + grid_offset_y
            z = 0
            center = (x, y, z)
            faces = prism_faces(center, R, H)
            # For each face keep only those requested in visible_faces
            for key, verts3d in faces:
                if key not in visible_faces:
                    continue
                avg_z = float(np.mean(verts3d[:, 2]))
                avg_x = float(np.mean(verts3d[:, 0]))
                avg_y = float(np.mean(verts3d[:, 1]))
                # Store enough info to draw later
                all_faces.append({
                    "key": key,
                    "verts3d": verts3d,
                    "avg_z": avg_z,
                    "avg_xy": avg_x + avg_y  # tie-breaker
                })

    # Sort faces by depth: draw from far (small avg_z) to near (large avg_z)
    all_faces.sort(key=lambda f: (f["avg_z"], f["avg_xy"]))

    # Draw sorted faces
    for face in all_faces:
        key = face["key"]
        verts3d = face["verts3d"]
        projected = [project_3d_to_2d(v) for v in verts3d]

        # If we have an image for this face, texture-map it clipped to the polygon
        img = face_images.get(key)
        if img is not None:
            xs = [p[0] for p in projected]
            ys = [p[1] for p in projected]
            minx = int(math.floor(min(xs)))
            miny = int(math.floor(min(ys)))
            maxx = int(math.ceil(max(xs)))
            maxy = int(math.ceil(max(ys)))
            pad = 3
            minx -= pad
            miny -= pad
            maxx += pad
            maxy += pad
            w = max(1, maxx - minx)
            h = max(1, maxy - miny)

            # create surfaces sized to bounding rect
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
            # Choose color per face type as fallback
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

        # Optionally label side faces (small, for debugging)
        if isinstance(key, int):
            cx = sum(p[0] for p in projected) / len(projected)
            cy = sum(p[1] for p in projected) / len(projected)
            # Write the face index in white
            # label = font.render(str(key), True, WHITE)
            # screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))


def main():
    running = True
    R = 30
    H = 60
    # Which faces to show per prism. Keep top + the three visible side faces.
    visible_faces = ["top", 0, 1, 5]

    # Example: explicitly set number of columns and rows in the grid.
    # Change these two values to adjust the number of tiles drawn.
    example_cols = 30  # e.g. 12 to force 12 columns
    example_rows = 26  # e.g. 8 to force 8 rows

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        # Build mapping for exported face images (paths relative to repository)
        face_dir = os.path.join(os.path.dirname(__file__), "exported_faces")
        face_image_paths = {
            "top": os.path.join(face_dir, "onexit_top.png"),
            0: os.path.join(face_dir, "onexit_0.png"),
            1: os.path.join(face_dir, "onexit_1.png"),
            5: os.path.join(face_dir, "onexit_5.png"),
        }

        draw_tiled_prisms(screen, R=R, H=H, visible_faces=visible_faces, cols=example_cols, rows=example_rows, face_image_paths=face_image_paths)
        # draw_prism(screen, center=(0, 0, 0), R=R, H=H, visible_faces=visible_faces, face_image_paths=face_image_paths)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

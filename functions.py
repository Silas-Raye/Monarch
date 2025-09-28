import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import random
import numpy as np
import pandas as pd
import math
import csv
from setup import game_setup

# Access the setup variables from the imported dictionary
screen = game_setup["screen"]
font = game_setup["font"]
WIDTH = game_setup["WIDTH"]
HEIGHT = game_setup["HEIGHT"]
CENTER_X = game_setup["CENTER_X"]
CENTER_Y = game_setup["CENTER_Y"]
BLACK = game_setup["BLACK"]
WHITE = game_setup["WHITE"]
SIDE_FACE_COLOR = game_setup["SIDE_FACE_COLOR"]
TOP_FACE_COLOR = game_setup["TOP_FACE_COLOR"]
iso_matrix = game_setup["iso_matrix"]
cell_radius = game_setup["cell_radius"]
cell_height = game_setup["cell_height"]
num_cols = game_setup["num_cols"]
num_rows = game_setup["num_rows"]

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

def draw_tiled_prisms(screen, R=cell_radius, H=cell_height, cols=num_cols, rows=num_rows, margin=6, map_dir = "exported_maps", viz_map=None, h_map=None, face_cache=None, labs=False):
    """
    Tile the screen with hexagonal prisms and draw them in hex-diagonal order using a pre-rendered cache.

    Args:
        screen (pygame.Surface): The surface to draw on.
        R (int): The radius of the hexagonal cell.
        H (int): The height of the prism.
        cols (int): The number of columns in the grid.
        rows (int): The number of rows in the grid.
        margin (int): Margin for the grid size.
        map_dir (str): Directory for map files.
        viz_map (str or pd.DataFrame): Visibility map.
        h_map (str or pd.DataFrame): Height map.
        face_cache (dict): A dictionary of pre-rendered face surfaces.
        labs (bool): If True, draw labels on the prisms.
    """
    ACCEPTABLE_TYPES = (str, pd.DataFrame)
    if (not isinstance(viz_map, ACCEPTABLE_TYPES) and
        not isinstance(h_map, ACCEPTABLE_TYPES)):
        print("Error: viz_map and h_map must be either a filename (str) or a pandas DataFrame.")

    if isinstance(viz_map, pd.DataFrame):
        pass
    elif isinstance(viz_map, str):
        os.makedirs(map_dir, exist_ok=True)
        viz_map_path = os.path.join(map_dir, viz_map) if viz_map else None

    if isinstance(h_map, pd.DataFrame):
        pass
    elif isinstance(h_map, str):
        os.makedirs(map_dir, exist_ok=True)
        h_map_path = os.path.join(map_dir, viz_map) if viz_map else None

    def offset_oddq_to_axial(r, c):
        aq = c
        ar = r - ((c - (c & 1)) // 2)
        return aq, ar

    def diagonal_band_index(r, c):
        aq, ar = offset_oddq_to_axial(r, c)
        return aq + ar

    if face_cache:
        visible_faces = set(face_cache.keys())
    else:
        visible_faces = set(["top", 0, 1, 5])

    dx = 1.5 * R
    dy = math.sqrt(3) * R

    if cols is None:
        cols = int(WIDTH / dx) + margin
    if rows is None:
        rows = int(HEIGHT / dy) + margin

    grid_offset_x = -cols * dx / 2
    grid_offset_y = -rows * dy / 2

    if isinstance(viz_map, pd.DataFrame):
        pass
    elif isinstance(viz_map, str):
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

    if isinstance(viz_map, pd.DataFrame):
        pass
    elif isinstance(viz_map, str):
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

    local_face_order = ["bottom", 2, 3, 4, 1, 0, 5, "top"]
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
            cells.append((b, c, r))
    cells.sort(key=lambda t: (t[0], t[1], t[2]))

    for _, c, r in cells:
        x = c * dx + grid_offset_x
        y = r * dy + (c % 2) * (dy / 2) + grid_offset_y
        z = 0

        try:
            h_offset = int(h_map[r][c]) if h_map is not None else 0
        except Exception:
            h_offset = 0

        center = (x, y, z)
        projected_center = project_3d_to_2d(center)
        
        # Only get the faces for drawing labs
        faces_list = prism_faces(center, R, H)
        faces_dict = {key: verts for key, verts in faces_list}
        
        for key in local_face_order:
            if key not in visible_faces:
                continue

            if face_cache and key in face_cache:
                face_data = face_cache[key]
                face_surf = face_data["surface"]
                face_offset = face_data["offset"]
                
                # Blit the pre-rendered surface
                screen.blit(face_surf, (
                    projected_center[0] + face_offset[0],
                    projected_center[1] + face_offset[1] - h_offset
                ))

            else:
                # Fallback to drawing without cache
                verts3d = faces_dict[key]
                projected = []
                for v in verts3d:
                    px, py = project_3d_to_2d(v)
                    projected.append((px, py - h_offset))
                
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
                if key == "top":
                    verts3d = faces_dict[key]
                    projected = [project_3d_to_2d(v) for v in verts3d]
                    
                    cx = sum(p[0] for p in projected) / len(projected)
                    cy = sum(p[1] for p in projected) / len(projected)

                    label_txt = f"{r+1},{c+1}"
                    label_surf = font.render(label_txt, True, (255, 255, 255))
                    shadow = font.render(label_txt, True, (0, 0, 0))
                    lx = int(cx - label_surf.get_width() // 2)
                    ly = int(cy - label_surf.get_height() // 2) - 2 - h_offset
                    screen.blit(shadow, (lx + 1, ly + 1))
                    screen.blit(label_surf, (lx, ly))

def get_face_color(key):
    """Returns the color for a given face key."""
    if key == "top":
        return TOP_FACE_COLOR
    else: # All side faces
        return SIDE_FACE_COLOR

def save_visible_faces(visible_faces, vertices, faces, project_fn, out_dir=os.path.join("assets", "exported_faces"), prefix="face", outline=False, outline_color=BLACK):
    """Save each face in visible_faces to a tightly-cropped image.

    Produces one files per face: PNG with alpha (best for Photoshop).
    """
    os.makedirs(out_dir, exist_ok=True)

    for key in visible_faces:
        face_indices = faces[key]
        face_vertices = vertices[face_indices]
        projected = [project_fn(v) for v in face_vertices]

        xs = [p[0] for p in projected]
        ys = [p[1] for p in projected]
        min_x, max_x = int(min(xs)), int(max(xs))
        min_y, max_y = int(min(ys)), int(max(ys))

        pad = 4
        w = max(1, max_x - min_x + 2 * pad)
        h = max(1, max_y - min_y + 2 * pad)

        # Local polygon coords
        local_poly = [(int(p[0]) - min_x + pad, int(p[1]) - min_y + pad) for p in projected]

        # Get the color for the face
        face_color_rgb = get_face_color(key)
        # Create a Pygame surface with alpha channel for the PNG
        surf_png = pygame.Surface((w, h), pygame.SRCALPHA)
        surf_png.fill((0, 0, 0, 0)) # Fill with transparent
        pygame.draw.polygon(surf_png, face_color_rgb + (255,), local_poly)
        if outline:
            pygame.draw.polygon(surf_png, outline_color + (255,), local_poly, 2)
        # Optional: Add a black border to the saved PNG
        # pygame.draw.polygon(surf_png, (0, 0, 0, 255), local_poly, 2)

        name_key = str(key)
        if prefix:
            base = f"{prefix}_{name_key}"
        else:
            base = f"face_{name_key}"

        png_path = os.path.join(out_dir, base + ".png")

        try:
            pygame.image.save(surf_png, png_path)
            print(f"Saved: {png_path}")
        except Exception as e:
            print(f"Failed saving {base}: {e}")

def quick_map(map_rows=num_rows, map_cols=num_cols, default_value=0, h_randomize=False, viz_randomize=False, random_min=None, random_max=None, seed=42, map_name=None):
    """
    Creates a pandas DataFrame with a specified number of rows and columns,
    populating each cell with a default value or a random integer.
    
    Args:
        map_rows (int): The number of rows for the DataFrame.
        map_cols (int): The number of columns for the DataFrame.
        default_value (int): The default value to fill the DataFrame with.
        h_randomize (bool): If True, fills the entire DataFrame with random integers.
        viz_randomize (bool): If True, fills the DataFrame with zeros and then randomly sets `random_max` cells to 1.
        random_min (int): The minimum value for random integers when h_randomize is True.
        random_max (int): The maximum value for random integers when h_randomize is True, or the number of cells to set to 1 when viz_randomize is True.
        seed (int): The seed for the random number generator.
        map_name (str): The name of the CSV file to save the DataFrame to.

    Returns:
        pd.DataFrame: The generated pandas DataFrame.
    """
    if seed != 42:
        random.seed(seed)

    data = []

    if h_randomize:
        if random_min is None or random_max is None:
            raise ValueError("h_randomize requires both random_min and random_max to be set.")
        data = [[random.randint(random_min, random_max) for _ in range(map_cols)] for _ in range(map_rows)]
    
    elif viz_randomize:
        if random_max is None:
            raise ValueError("viz_randomize requires random_max to be set.")
        
        # Start with a grid of default_value
        data = [[default_value for _ in range(map_cols)] for _ in range(map_rows)]
        
        # Calculate total number of cells
        total_cells = map_rows * map_cols
        
        # Number of cells to set to 1
        num_ones = min(random_max, total_cells)
        
        # Generate all possible coordinates
        all_coords = [(r, c) for r in range(map_rows) for c in range(map_cols)]
        
        # Select a random sample of coordinates to set to 1
        random_coords = random.sample(all_coords, num_ones)
        
        # Update the data grid
        for r, c in random_coords:
            data[r][c] = 1
            
    else:
        data = [[default_value for _ in range(map_cols)] for _ in range(map_rows)]

    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file if map_name is provided
    exported_faces = os.path.join("assets", "exported_maps")
    if map_name is not None:
        os.makedirs(exported_faces, exist_ok=True)
        map_path = os.path.join(exported_faces, map_name)
        df.to_csv(map_path, index=False, header=False)
    
    return df

def map_add(io_dir = os.path.join("assets", "exported_maps"), path1_name = None, path2_name = None, output_name = "map_sum.csv"):
    """
    Adds corresponding elements of two CSV files and saves the result to a new CSV.

    Args:
        path1 (str): The file path of the first CSV.
        path2 (str): The file path of the second CSV.
        output_name (str): The name of the output CSV file. Defaults to "map_sum.csv".
    """

    # Ensure the directory exists
    os.makedirs(io_dir, exist_ok=True)
    path1 = os.path.join(io_dir, path1_name)
    path2 = os.path.join(io_dir, path2_name)
    output_path = os.path.join(io_dir, output_name)

    try:
        # Read the CSV files into pandas DataFrames
        df1 = pd.read_csv(path1, header=None)
        df2 = pd.read_csv(path2, header=None)
        
        # Check if the DataFrames have the same shape (rows and columns)
        if df1.shape != df2.shape:
            print("Error: The CSV files do not have the same number of rows and columns.")
            return
        
        # Add the corresponding elements of the two DataFrames
        df_sum = df1.add(df2)
        
        # Save the resulting DataFrame to a new CSV file
        df_sum.to_csv(output_path, index=False, header=None)
        print(f"Success! The sum of the CSVs has been saved to '{output_path}'.")

    except FileNotFoundError:
        print("Error: One or both of the specified CSV files were not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def create_prism_cache(face_image_paths):
    """
    Pre-renders and caches textured prism faces.

    Args:
        face_image_paths (dict): A dictionary mapping face keys to image paths.

    Returns:
        dict: A dictionary mapping face keys to pre-rendered Pygame Surfaces.
    """
    face_images = load_face_images(face_image_paths)
    face_cache = {}

    # Define a canonical prism for pre-rendering
    center = (0, 0, 0)
    faces_list = prism_faces(center, cell_radius, cell_height)
    faces_dict = {key: verts for key, verts in faces_list}

    # Pre-render each visible face
    for key, verts3d in faces_dict.items():
        img = face_images.get(key)
        if img is not None:
            projected = [project_3d_to_2d(v) for v in verts3d]
            xs = [p[0] for p in projected]
            ys = [p[1] for p in projected]
            minx = int(math.floor(min(xs)))
            miny = int(math.floor(min(ys)))
            maxx = int(math.ceil(max(xs)))
            maxy = int(math.ceil(max(ys)))
            pad = 4
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

            face_cache[key] = {
                "surface": image_surf,
                "offset": (minx - CENTER_X, miny - CENTER_Y) # Store the offset relative to the center
            }

    return face_cache

import pygame
import sys
import math

def iso_project(x, y, z=0):
	"""Project 3D (x,y,z) coordinates to 2D isometric screen coordinates.

	Uses a simple isometric projection where x and y are ground plane axes and z is height.
	"""
	# scale factors (controls cube size)
	sx = 1
	sy = 0.5
	screen_x = (x - y) * sx
	screen_y = (x + y) * sy - z
	return int(screen_x), int(screen_y)

def draw_cube(x, y, h, side_path, top_path):
    # Signature: draw_cube(x, y, h, side_path, top_path)
    # Treat 1 tile == cell_pixels world units
    size = cell_pixels
    # If h > 0: cube sits on top of the grid (bottom at z=0, top at z = h*size)
    # If h < 0: cube extends below the grid (top at z=0, bottom at z = h*size)
    if h == 0:
        # nothing to draw for zero height
        return
    if h > 0:
        top_z = h * size
        bottom_z = 0
    else:
        top_z = 0
        bottom_z = h * size  # negative

    # World coordinates of the four corners of the tile footprint
    wx = x * size
    wy = y * size
    # order: top-left, top-right, bottom-right, bottom-left (in world x/y)
    top_world = [
        (wx, wy, top_z),
        (wx + size, wy, top_z),
        (wx + size, wy + size, top_z),
        (wx, wy + size, top_z),
    ]
    bottom_world = [
        (wx, wy, bottom_z),
        (wx + size, wy, bottom_z),
        (wx + size, wy + size, bottom_z),
        (wx, wy + size, bottom_z),
    ]

    # Project to screen coordinates and apply offset
    def proj(p):
        sx, sy = iso_project(p[0], p[1], p[2])
        return (offset[0] + sx, offset[1] + sy)

    top_pts = [proj(p) for p in top_world]
    bottom_pts = [proj(p) for p in bottom_world]

    # Front (south) side: connect top_pts[3], top_pts[2], bottom_pts[2], bottom_pts[3]
    front_poly = [top_pts[3], top_pts[2], bottom_pts[2], bottom_pts[3]]
    # Right (east) side: connect top_pts[1], top_pts[2], bottom_pts[2], bottom_pts[1]
    right_poly = [top_pts[1], top_pts[2], bottom_pts[2], bottom_pts[1]]

    # Helper: texture loading cache (store surfaces indexed by path)
    if not hasattr(draw_cube, '_tex_cache'):
        draw_cube._tex_cache = {}

    def load_texture(path):
        if path in draw_cube._tex_cache:
            return draw_cube._tex_cache[path]
        # Try to load image, return a surface (or None)
        try:
            img = pygame.image.load(path).convert_alpha()
            draw_cube._tex_cache[path] = img
            return img
        except Exception:
            # fallback: a 1x1 white surface so tiling still works
            fallback = pygame.Surface((1, 1), pygame.SRCALPHA)
            fallback.fill((200, 200, 200, 255))
            draw_cube._tex_cache[path] = fallback
            return fallback
    
    # Load textures
    side_tex_source = load_texture(side_path)
    top_tex_source = load_texture(top_path)

    # --- NEW IMPLEMENTATION: Stretch textures to fit polygons ---
    
    # Front side
    min_x_front = min(p[0] for p in front_poly)
    min_y_front = min(p[1] for p in front_poly)
    max_x_front = max(p[0] for p in front_poly)
    max_y_front = max(p[1] for p in front_poly)
    w_front = max(1, int(math.ceil(max_x_front - min_x_front)))
    h_front = max(1, int(math.ceil(max_y_front - min_y_front)))
    
    try:
        side_tex_scaled = pygame.transform.smoothscale(side_tex_source, (w_front, h_front))
    except Exception:
        side_tex_scaled = pygame.transform.scale(side_tex_source, (w_front, h_front))

    front_surface = pygame.Surface((w_front, h_front), pygame.SRCALPHA)
    front_surface.blit(side_tex_scaled, (0, 0))
    mask_front = pygame.Surface((w_front, h_front), pygame.SRCALPHA)
    shifted_front = [(p[0] - min_x_front, p[1] - min_y_front) for p in front_poly]
    pygame.draw.polygon(mask_front, (255, 255, 255, 255), shifted_front)
    front_surface.blit(mask_front, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    screen.blit(front_surface, (min_x_front, min_y_front))

    # Right side
    min_x_right = min(p[0] for p in right_poly)
    min_y_right = min(p[1] for p in right_poly)
    max_x_right = max(p[0] for p in right_poly)
    max_y_right = max(p[1] for p in right_poly)
    w_right = max(1, int(math.ceil(max_x_right - min_x_right)))
    h_right = max(1, int(math.ceil(max_y_right - min_y_right)))

    # Darken the right side texture
    try:
        right_tex_scaled = pygame.transform.smoothscale(side_tex_source, (w_right, h_right))
    except Exception:
        right_tex_scaled = pygame.transform.scale(side_tex_source, (w_right, h_right))

    # Darken percentage (0.0 = completely black, 1.0 = no darkening)
    darken_pct = 0.5  # CHANGE ME

    # Compute color multiplier based on darken_pct
    dark_val = int(255 * darken_pct)
    darken = pygame.Surface(right_tex_scaled.get_size(), pygame.SRCALPHA)
    darken.fill((dark_val, dark_val, dark_val, 255))

    # Apply darkening
    right_tex_scaled.blit(darken, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    # Create surface to hold final masked result
    right_surface = pygame.Surface((w_right, h_right), pygame.SRCALPHA)
    right_surface.blit(right_tex_scaled, (0, 0))

    # Build polygon mask
    mask_right = pygame.Surface((w_right, h_right), pygame.SRCALPHA)
    shifted_right = [(p[0] - min_x_right, p[1] - min_y_right) for p in right_poly]
    pygame.draw.polygon(mask_right, (255, 255, 255, 255), shifted_right)

    # Apply mask
    right_surface.blit(mask_right, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    # Draw final surface onto screen
    screen.blit(right_surface, (min_x_right, min_y_right))

    # Top face
    min_x_top = min(p[0] for p in top_pts)
    min_y_top = min(p[1] for p in top_pts)
    max_x_top = max(p[0] for p in top_pts)
    max_y_top = max(p[1] for p in top_pts)
    w_top = max(1, int(math.ceil(max_x_top - min_x_top)))
    h_top = max(1, int(math.ceil(max_y_top - min_y_top)))

    try:
        top_tex_scaled = pygame.transform.smoothscale(top_tex_source, (w_top, h_top))
    except Exception:
        top_tex_scaled = pygame.transform.scale(top_tex_source, (w_top, h_top))
    
    top_surface = pygame.Surface((w_top, h_top), pygame.SRCALPHA)
    top_surface.blit(top_tex_scaled, (0, 0))
    mask_top = pygame.Surface((w_top, h_top), pygame.SRCALPHA)
    shifted_top = [(p[0] - min_x_top, p[1] - min_y_top) for p in top_pts]
    pygame.draw.polygon(mask_top, (255, 255, 255, 255), shifted_top)
    top_surface.blit(mask_top, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    screen.blit(top_surface, (min_x_top, min_y_top))

    # --- End of new implementation ---

    # Draw outlines for clarity (on top of fills)
    edge_color = (10, 10, 10)
    pygame.draw.polygon(screen, edge_color, front_poly, 1)
    pygame.draw.polygon(screen, edge_color, right_poly, 1)
    pygame.draw.polygon(screen, edge_color, top_pts, 1)

pygame.init()
screen_width = 960
screen_height = 640
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Isometric Cubes - Testbench")
clock = pygame.time.Clock()

running = True

# Grid parameters
grid_size = 17 # gets subtracted by one later
cell_pixels = 30

# Compute offset to center the whole grid on screen
# Grid spans from 0..(grid_size-1) in both axes; compute projected bounds
corners = [iso_project(0, 0), iso_project((grid_size - 1) * cell_pixels, 0), iso_project(0, (grid_size - 1) * cell_pixels), iso_project((grid_size - 1) * cell_pixels, (grid_size - 1) * cell_pixels)]
xs = [c[0] for c in corners]
ys = [c[1] for c in corners]
grid_width = max(xs) - min(xs)
grid_height = max(ys) - min(ys)
center = (screen_width // 2, screen_height // 2 - 30)
offset = (center[0] - grid_width // 2 - min(xs), center[1] - grid_height // 2 - min(ys))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    screen.fill((30, 30, 40))

    # Draw a subtle ground grid for context (16x16)
    line_color = (40, 40, 50)
    for i in range(grid_size):
        # vertical-ish grid lines along y
        start = iso_project(i * cell_pixels, 0)
        end = iso_project(i * cell_pixels, (grid_size - 1) * cell_pixels)
        pygame.draw.line(screen, line_color, (offset[0] + start[0], offset[1] + start[1]), (offset[0] + end[0], offset[1] + end[1]), 1)
    for j in range(grid_size):
        # horizontal-ish grid lines along x
        start = iso_project(0, j * cell_pixels)
        end = iso_project((grid_size - 1) * cell_pixels, j * cell_pixels)
        pygame.draw.line(screen, line_color, (offset[0] + start[0], offset[1] + start[1]), (offset[0] + end[0], offset[1] + end[1]), 1)

    # Demo: draw some cubes for verification
    side_path = "grass_side_carried.png"
    top_path = "grass_carried.png"
    side_path2 = "tree.jpg"
    
    draw_cube(0, 13, 0.4, "cloud.jpg", top_path)
    draw_cube(0, 14, 0.2, "cloud.jpg", top_path)
    draw_cube(0, 15, 0.1, "cloud.jpg", top_path)
    draw_cube(1, 14, 0.4, "cloud.jpg", top_path)
    draw_cube(1, 15, 0.2, "cloud.jpg", top_path)
    draw_cube(2, 15, 0.4, "cloud.jpg", top_path)
    
    # taller cube at (8,6)
    draw_cube(8, 6, 3, side_path2, top_path)

    # negative (below ground) cube at (10,10)
    draw_cube(10, 10, -1, side_path, top_path)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()

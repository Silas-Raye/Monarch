import pygame
import sys
import math
import random

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
screen_width = 992
screen_height = 640
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Isometric Cubes - Testbench")
clock = pygame.time.Clock()

running = True

# Grid parameters
grid_size = 45 # gets subtracted by one later
cell_pixels = 30

# Compute offset to center the whole grid on screen
# Grid spans from 0..(grid_size-1) in both axes; compute projected bounds
corners = [iso_project(0, 0), iso_project((grid_size - 1) * cell_pixels, 0), iso_project(0, (grid_size - 1) * cell_pixels), iso_project((grid_size - 1) * cell_pixels, (grid_size - 1) * cell_pixels)]
xs = [c[0] for c in corners]
ys = [c[1] for c in corners]
grid_width = max(xs) - min(xs)
grid_height = max(ys) - min(ys)
center = (screen_width // 2, screen_height // 2 - 30)
# Vertical offset (pixels) to shift the entire background grid and drawn cubes
# downward. Change this number to move the scene up/down without touching
# individual drawing code.
vert_offset = 80

# Offset to add to projected coordinates when drawing (includes vert_offset)
offset = (center[0] - grid_width // 2 - min(xs), center[1] - grid_height // 2 - min(ys) + vert_offset)

# Ground textures (use these for the full-grid demo)
ground_side_texture = "test-assets/dirt.png"
ground_top_texture = "test-assets/dirt_path_top.png"

# Seed RNG and pre-generate a height for every grid cell so heights remain
# stable across frames. Change rng_seed to get a different but repeatable
# layout.
rng_seed = 17
random.seed(rng_seed)
heights = [[random.uniform(0.1, 0.4) for x in range(grid_size)] for y in range(grid_size)]

# --- Sweet berry bush animated sprite setup ---------------------------------
# Load frames sweet_berry_bush_stage0.png .. sweet_berry_bush_stage4.png
def load_image_safe(path, fallback_size=(32, 48)):
    try:
        return pygame.image.load(path).convert_alpha()
    except Exception:
        surf = pygame.Surface(fallback_size, pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        pygame.draw.rect(surf, (180, 60, 60), surf.get_rect(), border_radius=6)
        return surf

bush_frames = []
for i in range(4):
    p = f"test-assets/sweet_berry_bush_stage{i}.png"
    bush_frames.append(load_image_safe(p, fallback_size=(cell_pixels, int(cell_pixels * 1.5))))

bush_frame_count = len(bush_frames)
# milliseconds per frame
bush_frame_duration = 200
# movement timing (ms)
bush_move_duration = 2000
# world (tile) start and end positions (x, y) in tile coordinates
bush_world_start = (20, 45)
bush_world_end = (20, 28)
# start time
bush_start_time = pygame.time.get_ticks()
# ---------------------------------------------------------------------------

# Typing speed override: if set to a number (chars/sec) the text_box will use
# this rate instead of interpolating from stime->etime. Set to None to keep
# the original behavior where stime..etime defines the duration.
text_typing_speed_chars_per_sec = 10

# Draw a simple typing textbox at the bottom of the screen.
def text_box(text, stime, etime):
    """Draw a white rectangle on the bottom 1/8th of the screen and type the
    provided text one character at a time between stime and etime (ms).
    If current time is outside [stime, etime] the box is not drawn.
    """
    now = pygame.time.get_ticks()
    if now < stime or now > etime:
        return

    # Determine how many characters to show.
    total_chars = len(text)
    if text_typing_speed_chars_per_sec is not None:
        # chars based on constant speed (chars/sec)
        elapsed = max(0, now - stime)
        chars = int((elapsed / 1000.0) * float(text_typing_speed_chars_per_sec))
        chars = max(0, min(total_chars, chars))
        visible = text[:chars]
    else:
        # clamp duration to avoid div-by-zero
        duration = max(1, etime - stime)
        progress = float(now - stime) / float(duration)
        # Determine how many characters to show (floor), ensure not negative
        chars = int(progress * total_chars)
        chars = max(0, min(total_chars, chars))
        visible = text[:chars]

    # Rectangle geometry (bottom 1/8th of screen)
    rect_h = screen_height // 8
    # Make the box a little smaller than full-screen so a border of the game
    # remains visible around it. Margin is proportional to screen width but
    # has a sensible minimum.
    margin = max(12, int(screen_width * 0.03))
    # lift box slightly so the margin also shows below the box
    rect_y = screen_height - rect_h - (margin // 2)
    rect = pygame.Rect(margin, rect_y, screen_width - 2 * margin, rect_h)

    # Draw white background and black border
    pygame.draw.rect(screen, (255, 255, 255), rect)
    pygame.draw.rect(screen, (0, 0, 0), rect, 2)

    # Render text and center it horizontally (fall back to left padding if
    # the text is wider than the available area). Vertically center always.
    font_size = max(12, rect_h // 3)
    font = pygame.font.Font(None, font_size)
    text_surf = font.render(visible, True, (10, 10, 10))
    padding = 10
    ty = rect.y + (rect.height - text_surf.get_height()) // 2

    max_text_width = rect.width - 2 * padding
    if text_surf.get_width() <= max_text_width:
        # Center horizontally
        tx = rect.x + (rect.width - text_surf.get_width()) // 2
        screen.blit(text_surf, (tx, ty))
    else:
        # Too wide: left-align with padding and clip the drawn area so it
        # doesn't overflow the box.
        tx = rect.x + padding
        clip_area = pygame.Rect(0, 0, max_text_width, text_surf.get_height())
        screen.blit(text_surf, (tx, ty), area=clip_area)


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

    # Draw a cube on every grid square using the precomputed heights so
    # the layout is stable and repeatable across frames.
    # Iterate rows (y) then columns (x) so tiles are drawn in back-to-front order
    # for correct overlap in isometric projection.
    for y in range(grid_size - 1):
        for x in range(grid_size - 1):
            h = heights[y][x]
            if ((x + 6 - grid_size//2) ** 2 + (y + 8 - grid_size//2) ** 2) <= (grid_size * 0.19) ** 2:
                draw_cube(x, y, 20, "test-assets/spruce_log.png", "test-assets/spruce_log.png")
            else:
                draw_cube(x, y, h, ground_side_texture, ground_top_texture)

    # draw_cube(5, 5, 3.1, "test-assets/spruce_log.png", "test-assets/spruce_log.png")

    # --- Animated sweet berry bush: update position and frame ----------
    now = pygame.time.get_ticks()
    elapsed = now - bush_start_time
    t = min(1.0, float(elapsed) / float(bush_move_duration))

    # linear interpolate world tile coordinates
    wx = bush_world_start[0] + (bush_world_end[0] - bush_world_start[0]) * t
    wy = bush_world_start[1] + (bush_world_end[1] - bush_world_start[1]) * t

    # choose animation frame (looping through frames while moving)
    frame_index = int((elapsed // bush_frame_duration) % bush_frame_count)
    frame_surf = bush_frames[frame_index]

    # Project the world tile origin (top-left of tile) in pixels
    # We want the bush to sit centered on the tile; assume sprite bottom-center aligns to tile's top surface center
    # Compute tile center in world coords (use center of tile footprint)
    tile_center_x = (wx + 0.5) * cell_pixels
    tile_center_y = (wy + 0.5) * cell_pixels
    screen_pos = iso_project(tile_center_x, tile_center_y, 0)
    screen_x = offset[0] + screen_pos[0]
    screen_y = offset[1] + screen_pos[1]

    # Adjust so sprite bottom-center sits on the tile (sprite origin bottom center)
    sw, sh = frame_surf.get_size()
    blit_x = int(screen_x - sw // 2)
    # lift the sprite up a bit so it appears on top of the tile (account for sprite height)
    blit_y = int(screen_y - sh)

    screen.blit(frame_surf, (blit_x, blit_y))
    # --------------------------------------------------------------------

    # Test the text_box: show "what a nice tree" from 0 ms to 300 ms
    text_box("what a nice tree", 3000, 12000)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()

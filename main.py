import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from functions import draw_tiled_prisms, quick_map
from setup import game_setup
from player import Player

# Access the setup variables from the imported dictionary
screen = game_setup["screen"]
clock = game_setup["clock"]
CENTER_X = game_setup["CENTER_X"]
CENTER_Y = game_setup["CENTER_Y"]
BLACK = game_setup["BLACK"]
WHITE = game_setup["WHITE"]
cell_height = game_setup["cell_height"]

def main():
    # Player initialization
    player_x = CENTER_X
    player_y = CENTER_Y

    # Ant sprite sheet details
    ant_sprite_sheet_path = "assets/sprites/pc/ant_walk_sheet.png"
    sprite_width = 450
    sprite_height = 300
    num_sprites = 6

    player = Player(player_x, player_y, ant_sprite_sheet_path, sprite_width, sprite_height, num_sprites, scale=0.2)

    # Variables for movement
    move_direction = pygame.Vector2(0, 0)

    # Initialize font for FPS display
    pygame.font.init()
    font = pygame.font.Font(None, 36)
    
    # Create maps if they don't exist
    y0 = quick_map()
    y1 = quick_map(default_value=cell_height)
    all_viz = quick_map(default_value=1)
    y1_viz = quick_map(viz_randomize=True, random_max=10, seed=1)

    # Load face images and create cache
    face_dir = os.path.join("assets", "exported_faces")
    image_paths = {
        "top": os.path.join(face_dir, "face_top.png"),
        0: os.path.join(face_dir, "face_0.png"),
        1: os.path.join(face_dir, "face_1.png"),
        5: os.path.join(face_dir, "face_5.png"),
    }
    top_image_paths = {
        "top": os.path.join(face_dir, "face_top.png")
    }

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    move_direction.x = -1
                elif event.key == pygame.K_RIGHT:
                    move_direction.x = 1
                elif event.key == pygame.K_UP:
                    move_direction.y = -1
                elif event.key == pygame.K_DOWN:
                    move_direction.y = 1
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT and move_direction.x < 0:
                    move_direction.x = 0
                elif event.key == pygame.K_RIGHT and move_direction.x > 0:
                    move_direction.x = 0
                elif event.key == pygame.K_UP and move_direction.y < 0:
                    move_direction.y = 0
                elif event.key == pygame.K_DOWN and move_direction.y > 0:
                    move_direction.y = 0

        # Update player position based on movement vector
        if move_direction.length() > 0:
            player.move(move_direction.x, move_direction.y)
        else:
            player.stop_moving()

        screen.fill(BLACK)

        draw_tiled_prisms(screen, viz_map=all_viz, h_map=y0, face_image_paths=image_paths, labs=False)
        draw_tiled_prisms(screen, viz_map=y1_viz, h_map=y1, face_image_paths=image_paths, labs=False)
        
        # Update and draw the player
        player.update()
        player.draw(screen)

        # Get and display FPS
        fps = clock.get_fps()
        fps_text = font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

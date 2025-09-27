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
image_paths = game_setup["image_paths"]

def main():
    # Player initialization
    player_x = CENTER_X
    player_y = CENTER_Y

    # Ant sprite sheet details
    ant_sprite_sheet_path = "assets/sprites/pc/ant_walk_sheet.png"
    sprite_width = 450
    sprite_height = 300
    num_sprites = 6

    player = Player(player_x, player_y, ant_sprite_sheet_path, sprite_width, sprite_height, num_sprites)



    # Create maps if they don't exist
    y0 = quick_map()
    y1 = quick_map(default_value=cell_height)
    all_viz = quick_map(default_value=1)
    y1_viz = quick_map(viz_randomize=True, random_max=50)
    
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        draw_tiled_prisms(screen, viz_map=all_viz, h_map=y0, face_image_paths=image_paths, labs=False)
        draw_tiled_prisms(screen, viz_map=y1_viz, h_map=y1, face_image_paths=image_paths, labs=False)
        
        # Update and draw the player
        player.update()
        player.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

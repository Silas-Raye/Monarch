import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from functions import draw_tiled_prisms, create_map, map_add, quick_map
from setup import game_setup

# Access the setup variables from the imported dictionary
screen = game_setup["screen"]
clock = game_setup["clock"]
BLACK = game_setup["BLACK"]
WHITE = game_setup["WHITE"]
cell_height = game_setup["cell_height"]
image_paths = game_setup["image_paths"]

def main():
    # Create maps if they don't exist
    y0 = quick_map(default_value=0)
    y1 = quick_map(default_value=cell_height)
    y2 = quick_map(default_value=(cell_height*2))
    y3 = quick_map(default_value=(cell_height*3))
    y4 = quick_map(default_value=(cell_height*4))
    y0_viz = quick_map(default_value=1)
    
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        # draw_tiled_prisms(screen, viz_map=y0_viz, h_map=y0, face_image_paths=image_paths, labs=False)
        # draw_tiled_prisms(screen, viz_map="y1_viz.csv", h_map=y1, face_image_paths=image_paths, labs=False)
        # draw_tiled_prisms(screen, viz_map="y2_viz.csv", h_map=y2, face_image_paths=image_paths, labs=False)
        # draw_tiled_prisms(screen, viz_map="y2_viz.csv", h_map=y3, face_image_paths=image_paths, labs=False)
        # draw_tiled_prisms(screen, viz_map="y3_viz.csv", h_map=y4, face_image_paths=image_paths, labs=False)

        draw_tiled_prisms(screen, viz_map="y0_viz.csv", h_map=quick_map(randomize=True, random_min=-8, random_max=8, seed=8), face_image_paths=image_paths, labs=False)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

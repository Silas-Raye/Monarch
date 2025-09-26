import pygame
from functions import draw_tiled_prisms
from setup import game_setup

# Access the setup variables from the imported dictionary
screen = game_setup["screen"]
clock = game_setup["clock"]
BLACK = game_setup["BLACK"]
WHITE = game_setup["WHITE"]
image_paths = game_setup["image_paths"]
plant_paths = game_setup["plant_paths"]

def main():
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        draw_tiled_prisms(screen, viz_map="viz_map_unif.csv", h_map="h_map_rand.csv", face_image_paths=image_paths, labs=True)
        draw_tiled_prisms(screen, viz_map="viz_map_cust.csv", h_map="h_map_sum.csv", face_image_paths=plant_paths, labs=True)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

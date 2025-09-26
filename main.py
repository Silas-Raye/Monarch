import pygame
from functions import draw_tiled_prisms, create_map, map_add
from setup import game_setup

# Access the setup variables from the imported dictionary
screen = game_setup["screen"]
clock = game_setup["clock"]
BLACK = game_setup["BLACK"]
WHITE = game_setup["WHITE"]
image_paths = game_setup["image_paths"]
plant_paths = game_setup["plant_paths"]

def main():
    # Create maps if they don't exist
    create_map(file_name='h_map_rand.csv', randomize=True, random_min=0, random_max=20, seed=10)
    create_map(file_name='h_map_unif_30.csv', default_value=30)
    map_add(path1_name = 'h_map_rand.csv', path2_name = 'h_map_unif_30.csv', output_name = 'h_map_sum.csv')
    create_map(file_name='viz_map_unif.csv', default_value=1)
    # create_map(file_name='viz_map_cust.csv', default_value=0)
    
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        draw_tiled_prisms(screen, viz_map="viz_map_unif.csv", h_map="h_map_rand.csv", face_image_paths=image_paths, labs=False)
        draw_tiled_prisms(screen, viz_map="viz_map_cust.csv", h_map="h_map_sum.csv", face_image_paths=image_paths, labs=True)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

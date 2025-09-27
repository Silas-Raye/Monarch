import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

class Player:
    def __init__(self, x, y, sprite_sheet_path, frame_width, frame_height, num_frames, scale=1.0):
        """Initializes the Player."""
        self.x = x
        self.y = y
        self.moving = True
        self.frame_index = 0
        self.animation_speed = 5 # Higher is slower
        self.last_update = pygame.time.get_ticks()
        
        # Pass the scale factor to the load_sprites method
        self.frames = self.load_sprites(sprite_sheet_path, frame_width, frame_height, num_frames, scale)
        
        self.image = self.frames[self.frame_index]
        self.rect = self.image.get_rect(center=(x, y))

    def load_sprites(self, path, frame_width, frame_height, num_frames, scale):
        """Loads, slices, and scales the sprite sheet into individual frames."""
        sprite_sheet = pygame.image.load(path).convert()
        white = (255, 255, 255)
        sprite_sheet.set_colorkey(white)
        
        frames = []
        for i in range(num_frames):
            frame = pygame.Surface((frame_width, frame_height), pygame.SRCALPHA)
            frame.blit(sprite_sheet, (0, 0), (0, i * frame_height, frame_width, frame_height))
            
            # Scale the frame
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            scaled_frame = pygame.transform.scale(frame, (new_width, new_height))
            
            frames.append(scaled_frame)
        return frames

    def update(self):
        """Updates the player's animation."""
        now = pygame.time.get_ticks()
        if self.moving and now - self.last_update > self.animation_speed * 10:
            self.last_update = now
            self.frame_index = (self.frame_index + 1) % len(self.frames)
            self.image = self.frames[self.frame_index]
        elif not self.moving:
            self.frame_index = 0
            self.image = self.frames[self.frame_index]

    def draw(self, screen):
        """Draws the player on the screen."""
        screen.blit(self.image, self.rect)

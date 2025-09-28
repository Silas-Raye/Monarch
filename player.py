import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

class Player:
    def __init__(self, x, y, sprite_sheet_path, frame_width, frame_height, num_frames, scale=1.0):
        """Initializes the Player."""
        self.x = x
        self.y = y
        self.moving = False
        self.facing_right = True # New attribute to track direction
        self.speed = 5
        self.frame_index = 0
        self.animation_speed = 5 # Higher is slower
        self.last_update = pygame.time.get_ticks()

        # Load and store both normal and flipped sprites
        self.frames = self.load_sprites(sprite_sheet_path, frame_width, frame_height, num_frames, scale)
        self.flipped_frames = [pygame.transform.flip(frame, True, False) for frame in self.frames]

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
        """Updates the player's animation and position."""
        now = pygame.time.get_ticks()
        
        # Determine which set of frames to use based on direction
        current_frames = self.frames if self.facing_right else self.flipped_frames

        if self.moving and now - self.last_update > self.animation_speed * 10:
            self.last_update = now
            self.frame_index = (self.frame_index + 1) % len(current_frames)
            self.image = current_frames[self.frame_index]
        elif not self.moving:
            self.frame_index = 0
            self.image = current_frames[self.frame_index]

        # Update the rect to follow the new coordinates
        self.rect.center = (self.x, self.y)

    def move(self, dx, dy):
        """Moves the player and updates the facing direction."""
        self.x += dx * self.speed
        self.y += dy * self.speed
        self.moving = True

        # Update facing direction based on horizontal movement
        if dx > 0:
            self.facing_right = True
        elif dx < 0:
            self.facing_right = False

    def stop_moving(self):
        """Stops the player's movement and animation."""
        self.moving = False

    def draw(self, screen):
        """Draws the player on the screen."""
        screen.blit(self.image, self.rect)
        
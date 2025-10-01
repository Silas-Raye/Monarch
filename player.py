import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

class Player:
    def __init__(self, x, y, sprite_sheet_path, frame_width, frame_height, num_frames, scale=1.0):
        """Initializes the Player."""
        self.x = x
        self.y = y
        self.moving = False
        self.facing_right = True
        self.facing_up = False
        self.facing_down = False
        self.speed = 5
        self.frame_index = 0
        self.animation_speed = 5 # Higher is slower
        self.last_update = pygame.time.get_ticks()

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

            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            scaled_frame = pygame.transform.scale(frame, (new_width, new_height))

            frames.append(scaled_frame)
        return frames

    def update(self):
        """Updates the player's animation, position, and rotation."""
        now = pygame.time.get_ticks()
        
        current_frames = self.frames if self.facing_right else self.flipped_frames

        if self.moving and now - self.last_update > self.animation_speed * 10:
            self.last_update = now
            self.frame_index = (self.frame_index + 1) % len(current_frames)
        elif not self.moving:
            self.frame_index = 0

        base_image = current_frames[self.frame_index]
        
        # Apply rotation based on vertical movement, adjusting for horizontal direction
        if self.facing_up:
            if self.facing_right:
                self.image = pygame.transform.rotate(base_image, 30)
            else:
                self.image = pygame.transform.rotate(base_image, -30)
        elif self.facing_down:
            if self.facing_right:
                self.image = pygame.transform.rotate(base_image, -30)
            else:
                self.image = pygame.transform.rotate(base_image, 30)
        else:
            self.image = base_image

        self.rect = self.image.get_rect(center=(self.x, self.y))

    def move(self, dx, dy):
        """Moves the player and updates the facing direction and rotation state."""
        self.x += dx * self.speed
        self.y += dy * self.speed
        self.moving = True

        if dx > 0:
            self.facing_right = True
            self.facing_up = False
            self.facing_down = False
        elif dx < 0:
            self.facing_right = False
            self.facing_up = False
            self.facing_down = False
        
        if dy > 0:
            self.facing_down = True
            self.facing_up = False
        elif dy < 0:
            self.facing_up = True
            self.facing_down = False
            
        if dx != 0 and dy != 0:
            self.facing_up = False
            self.facing_down = False

    def stop_moving(self):
        """Stops the player's movement and animation."""
        self.moving = False
        self.facing_up = False
        self.facing_down = False

    def draw(self, screen):
        """Draws the player on the screen."""
        screen.blit(self.image, self.rect)

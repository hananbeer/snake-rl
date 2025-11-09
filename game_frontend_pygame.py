import pygame
import numpy as np
import time
from typing import Tuple, Optional
from game import SnakeGame, Vector

SPEED = 10

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 200, 0)
GREEN2 = (0, 100, 0)
GRAY = (100, 100, 100)


class SnakeGameFrontendPygame:
  def __init__(self, game: SnakeGame, block_size: int = 20, play_mode: str = 'absolute', tick_rate: float = 0, games: Optional[list] = None):
    self.game = game
    self.games = games if games is not None else [game]
    self.play_mode = play_mode
    self.block_size = block_size
    self.tick_rate = tick_rate

    pygame.init()
    self.font = pygame.font.SysFont('arial', 24)
    self.display = pygame.display.set_mode((game.width * block_size, game.height * block_size), pygame.RESIZABLE)
    pygame.display.set_caption('Snake')

  def resize(self, new_game: SnakeGame):
    """Resize the window when the game size changes."""
    self.game = new_game
    self.display = pygame.display.set_mode((new_game.width * self.block_size, new_game.height * self.block_size), pygame.RESIZABLE)


  def g2p(self, x: int, y: int) -> Tuple[int, int]:
    return x * self.block_size, y * self.block_size


  def render(self):
    self.display.fill(BLACK)

    if len(self.games) > 1:
      # Render average of multiple games
      self._render_average()
    else:
      # Render single game
      self._render_single()

    # Draw score (average if multiple games)
    if len(self.games) > 1:
      avg_score = sum(g.score for g in self.games) / len(self.games)
      avg_reward = sum(g.reward for g in self.games) / len(self.games)
      text = self.font.render(f"Avg Score: {avg_score:.1f} ({len(self.games)} agents)", True, WHITE)
      self.display.blit(text, [4, 4])
      text = self.font.render(f"Avg Reward: {avg_reward:.1f}", True, GREEN1 if avg_reward >= 0 else RED)
      self.display.blit(text, [4, 28])
    else:
      text = self.font.render(f"Score: {self.game.score}", True, WHITE)
      self.display.blit(text, [4, 4])
      text = self.font.render(f"Reward: {self.game.reward} ({'+' if self.game.step_reward > 0 else ''}{self.game.step_reward:.2f})", True, GREEN1 if self.game.reward >= 0 else RED)
      self.display.blit(text, [4, 28])

    if len(self.games) == 1 and self.game.is_game_over():
      text = self.font.render(f"GAME OVER", True, RED)
      self.display.blit(text, [self.game.width * self.block_size // 2 - text.get_width() // 2, self.game.height * self.block_size // 2 - text.get_height() // 2])

    pygame.display.flip()

    if self.tick_rate > 0:
      time.sleep(1 / self.tick_rate)

  def _render_single(self):
    # Draw snake (with rotation - head always centered, pointing up)
    for i, pt in enumerate(self.game.snake):
      color_range = [100, 255]
      lerp_factor = i / len(self.game.snake)
      color = [
        int(lerp_factor * (color_range[1] - color_range[0]) + color_range[0]),
        int(lerp_factor * (color_range[0] - color_range[1]) + color_range[1]),
        0
      ]
      pixel_x, pixel_y = self.g2p(pt.x, pt.y)
      block_rect = pygame.Rect(pixel_x, pixel_y, self.block_size, self.block_size)
      pygame.draw.rect(self.display, color, block_rect)
      inner_rect = pygame.Rect(pixel_x + self.block_size // 5, pixel_y + self.block_size // 5, self.block_size // 2, self.block_size // 2)
      color[0] *= 0.8
      color[1] *= 0.8
      pygame.draw.rect(self.display, color, inner_rect)

    # Draw food (with rotation)
    food_x, food_y = self.g2p(self.game.food.x, self.game.food.y)
    food_rect = pygame.Rect(food_x, food_y, self.block_size, self.block_size)
    pygame.draw.rect(self.display, RED, food_rect)

  def _render_average(self):
    # Create grids to track occupancy
    snake_grid = np.zeros((self.game.width, self.game.height), dtype=float)
    food_grid = np.zeros((self.game.width, self.game.height), dtype=float)
    
    # Count occurrences at each position
    for i, game in enumerate(self.games):
      # Count snake segments
      for i, pt in enumerate(game.snake):
        if 0 <= pt.x < self.game.width and 0 <= pt.y < self.game.height:
          # Weight head more than tail
          weight = 1.0 - (i / max(len(game.snake), 1)) * 0.5
          snake_grid[pt.x, pt.y] += weight
      
      # Count food
      if 0 <= game.food.x < self.game.width and 0 <= game.food.y < self.game.height:
        food_grid[game.food.x, game.food.y] += 1.0
    
    # Normalize by number of games
    max_snake = max(snake_grid.max(), 1.0)
    max_food = max(food_grid.max(), 1.0)
    
    # Draw averaged snake positions
    for x in range(self.game.width):
      for y in range(self.game.height):
        snake_count = snake_grid[x, y]
        if snake_count > 0:
          intensity = snake_count / max_snake
          # Lerp color from green to red based on snake index (use intensity as index, but clamp between 0 for head (green) and 1 for tail (red))
          # Green: (0, 200, 0)
          # Red: (200, 0, 0)
          lerp = intensity  # tail elements are lower intensity, so head is green, tail is red
          r = int((1 - lerp) * 50 + lerp * 200)
          g = int((1 - lerp) * 200 + lerp * 50)
          b = 30
          color = (r, g, b)
          pixel_x, pixel_y = self.g2p(x, y)
          block_rect = pygame.Rect(pixel_x, pixel_y, self.block_size, self.block_size)
          # Use alpha for transparency based on intensity
          s = pygame.Surface((self.block_size, self.block_size))
          # s.set_alpha(int(255 * intensity))
          s.fill(color)
          self.display.blit(s, (pixel_x, pixel_y))
    
    # Draw averaged food positions
    for x in range(self.game.width):
      for y in range(self.game.height):
        food_count = food_grid[x, y]
        if food_count > 0:
          intensity = food_count / max_food
          pixel_x, pixel_y = self.g2p(x, y)
          food_rect = pygame.Rect(pixel_x, pixel_y, self.block_size, self.block_size)
          s = pygame.Surface((self.block_size, self.block_size))
          s.set_alpha(int(255 * intensity))
          s.fill(RED)
          self.display.blit(s, (pixel_x, pixel_y))


  def render_text(self, text):
    # self.display.fill(BLACK)
    text = self.font.render(text, True, WHITE)
    self.display.blit(text, [self.game.width * self.block_size // 2 - text.get_width() // 2, self.game.height * self.block_size // 2 - text.get_height() // 2])
    pygame.display.flip()

  def handle_user_input(self):
    # Handle quit events
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()

      if event.type == pygame.VIDEORESIZE:
        # Handle window resize events
        self.display = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
        continue

      if event.type != pygame.KEYDOWN:
        continue

      if self.play_mode == 'absolute':
        if event.key == pygame.K_LEFT and self.game.direction != Vector(1, 0):
          self.game.direction = Vector(-1, 0)
        elif event.key == pygame.K_RIGHT and self.game.direction != Vector(-1, 0):
          self.game.direction = Vector(1, 0)
        elif event.key == pygame.K_UP and self.game.direction != Vector(0, 1):
          self.game.direction = Vector(0, -1)
        elif event.key == pygame.K_DOWN and self.game.direction != Vector(0, -1):
          self.game.direction = Vector(0, 1)
        else:
          continue
      elif self.play_mode == 'relative':
        if event.key == pygame.K_LEFT:
          self.game.direction = self.game.get_relative_direction('left')
        elif event.key == pygame.K_RIGHT:
          self.game.direction = self.game.get_relative_direction('right')
        else:
          continue
      elif self.play_mode == 'ai':
        raise NotImplementedError("AI keys mode not implemented")
      else:
        raise ValueError(f"Invalid keys mode: {self.play_mode}")

      return

  def _draw_walls(self):
    for x in range(self.width):
      # Top wall
      wall_x, wall_y = self.g2p(x, -1)
      wall_rect = pygame.Rect(wall_x, wall_y, self.block_size, self.block_size)
      pygame.draw.rect(self.display, GRAY, wall_rect)

      # Bottom wall
      wall_x, wall_y = self.g2p(x, self.height)
      wall_rect = pygame.Rect(wall_x, wall_y, self.block_size, self.block_size)
      pygame.draw.rect(self.display, GRAY, wall_rect)
    
    for y in range(self.height):
      # Left wall
      wall_x, wall_y = self.g2p(-1, y)
      wall_rect = pygame.Rect(wall_x, wall_y, self.block_size, self.block_size)
      pygame.draw.rect(self.display, GRAY, wall_rect)

      # Right wall
      wall_x, wall_y = self.g2p(self.width, y)
      wall_rect = pygame.Rect(wall_x, wall_y, self.block_size, self.block_size)
      pygame.draw.rect(self.display, GRAY, wall_rect)


if __name__ == '__main__':
  # TODO: size and tickrate from args, and render mode
  game = SnakeGame(43, 43)
  frontend = SnakeGameFrontendPygame(game, tick_rate=30, play_mode='absolute')

  # game loop
  while True:
    if frontend.handle_user_input():
      break
    
    reward, game_over, score = game.step()
    frontend.render()

    if game_over:
      print('Score', score)
      time.sleep(.75)
      game.reset()

  frontend.close()

import time
import math
import random
import pygame
from enum import Enum
from collections import namedtuple
from typing import Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt

Point = namedtuple('Point', 'x, y')
Vector = namedtuple('Vector', 'x, y')

class SnakeGame:
  def __init__(self, width: int, height: int, seed: Optional[int] = None):
    self.width = width
    self.height = height
    self.seed = seed
    self.rng = random.Random(seed) if seed is not None else random
    self.reset()


  @property
  def head(self) -> Point:
    return self.snake[0]


  def reset(self):
    self.direction = self.rng.choice([Vector(1, 0), Vector(0, 1), Vector(-1, 0), Vector(0, -1)])
    self.score = 0
    self.reward = 0
    self.step_reward = 0

    # Start in grid coordinates (center of grid)
    grid_x = self.rng.randint(2, self.width - 4)
    grid_y = self.rng.randint(2, self.height - 4)

    tail = Point(grid_x, grid_y)
    mid = self.get_next_position(tail, self.direction)
    head = self.get_next_position(mid, self.direction)
    self.snake = [
      head,
      mid,
      tail
    ]

    self.food = self.make_food()


  def make_food(self):
    max_attempts = self.width * self.height
    for _ in range(max_attempts):
      x = self.rng.randint(0, self.width - 1)
      y = self.rng.randint(0, self.height - 1)
      candidate = Point(x, y)
      if candidate not in self.snake:
        break
    
    return candidate


  def is_out_of_bounds(self, pt: Point):
    # Check boundary collision (in grid coordinates)
    return pt.x < 0 or pt.x >= self.width or pt.y < 0 or pt.y >= self.height


  def is_collision(self, pt: Point) -> bool:
    # Check self-collision
    return pt in self.snake[1:]


  def is_game_over(self, pt: Optional[Point] = None) -> bool:
    if pt is None:
      pt = self.head

    return self.is_out_of_bounds(pt) or self.is_collision(pt)


  def get_next_position(self, point: Point, dir: Vector) -> Point:
    return Point(point.x + dir.x, point.y + dir.y)


  def get_relative_direction(self, relative: str) -> Vector:
    if relative == 'forward':
      return self.direction
    elif relative == 'left':
      return Vector(self.direction.y, -self.direction.x)
    elif relative == 'right':
      return Vector(-self.direction.y, self.direction.x)
    else:
      raise ValueError(f"Invalid relative direction: {relative}")


  def step(self) -> Tuple[bool, int]:
    self.step_reward = 0

    # Get next head position
    next_head = self.get_next_position(self.head, self.direction)
    ate_food = (next_head == self.food)
    if not ate_food:
      self.snake.pop()
      if (next_head.x == self.food.x or next_head.y == self.food.y) and not (self.head.x == self.food.x or self.head.y == self.food.y):
        self.step_reward += 1
      elif not (next_head.x == self.food.x or next_head.y == self.food.y) and (self.head.x == self.food.x or self.head.y == self.food.y):
        self.step_reward -= 5

      # dist = math.sqrt((next_head.x - self.food.x) ** 2 + (next_head.y - self.food.y) ** 2)
      # self.step_reward -= dist - 5
    else:
      self.food = self.make_food()
      self.score += 1
      self.step_reward += 50

    self.snake.insert(0, next_head)

    is_game_over = self.is_game_over()
    if is_game_over:
      self.step_reward -= 20

    self.reward += self.step_reward
    return self.step_reward, is_game_over, self.score

class SnakeGameAI(SnakeGame):
  def world_view(self) -> np.ndarray:
    grid = np.zeros((self.width, self.height), dtype=int)
    for pt in self.snake:
      grid[pt.x, pt.y] = 1
    grid[self.food.x, self.food.y] = 2
    return grid

  def local_view(self, head: Point) -> np.ndarray:
    grid = np.zeros((self.block_size, self.block_size), dtype=int)

    # Define rotation matrix based on self.direction
    # Up (0,-1), Right (1,0), Down (0,1), Left (-1,0)
    # Reference (snake head always faces "up" in local view)
    direction = self.direction
    # Normalize
    dir_map = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
    rotate = dir_map.get((direction.x, direction.y), 0)

    # Function to rotate relative coordinates (dx, dy) according to self.direction
    def rotate_xy(dx, dy, rot):
      # rot: 0 = up, 1 = right, 2 = down, 3 = left (counterclockwise)
      for _ in range(rot):
        dx, dy = -dy, dx
      return dx, dy

    for pt in self.snake:
      dx = pt.x - head.x
      dy = pt.y - head.y
      rel_x, rel_y = rotate_xy(dx, dy, rotate)
      loc_x = rel_x + self.block_size // 2
      loc_y = rel_y + self.block_size // 2
      if 0 <= loc_x < self.block_size and 0 <= loc_y < self.block_size:
        grid[loc_x, loc_y] = 1

    return grid


SPEED = 10

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 200, 0)
GREEN2 = (0, 100, 0)
GRAY = (100, 100, 100)


class SnakeGameFrontend:
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


class SnakeGameMatplotlibFrontend:
  def __init__(self, game: SnakeGame, tick_rate: float = 0, games: Optional[List[SnakeGame]] = None, play_mode: str = 'absolute'):
    self.game = game
    self.games = games if games is not None else [game]
    self.tick_rate = tick_rate
    self.play_mode = play_mode
    self.key_pressed = None
    self.quit_requested = False
    
    # Create custom colormap: 0=black (empty), 1=green (snake), 2=red (food)
    from matplotlib.colors import ListedColormap
    colors = ['black', 'green', 'red']
    self.cmap = ListedColormap(colors)
    
    # Initialize matplotlib figure
    plt.ion()  # Turn on interactive mode
    self.fig, self.ax = plt.subplots(figsize=(8, 8))
    try:
      self.fig.canvas.set_window_title('Snake Game')
    except AttributeError:
      # Some backends don't support set_window_title
      pass
    self.ax.set_title('Snake Game', fontsize=16, fontweight='bold')
    self.ax.set_aspect('equal')
    self.ax.set_xticks([])
    self.ax.set_yticks([])
    plt.tight_layout()
    
    # Create initial empty grid for the image (will be updated with set_data)
    initial_grid = np.zeros((game.height, game.width), dtype=int)
    self.im = self.ax.imshow(initial_grid, cmap=self.cmap, vmin=0, vmax=2, 
                            interpolation='nearest', origin='upper', 
                            animated=True)
    
    # Disable autoscaling for better performance
    self.ax.set_autoscale_on(False)
    
    # Create text objects that we'll update instead of recreating
    self.info_text_obj = None
    self.game_over_text_obj = None
    
    # Connect keyboard events
    self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    self.fig.canvas.mpl_connect('close_event', self._on_close)
    
    plt.show(block=False)

  def resize(self, new_game: SnakeGame):
    """Resize when the game size changes."""
    old_width, old_height = self.game.width, self.game.height
    self.game = new_game
    # Recreate image if size changed
    if new_game.width != old_width or new_game.height != old_height:
      initial_grid = np.zeros((new_game.height, new_game.width), dtype=int)
      self.im.remove()
      self.im = self.ax.imshow(initial_grid, cmap=self.cmap, vmin=0, vmax=2, 
                              interpolation='nearest', origin='upper',
                              animated=True)
      # Reset text objects
      if self.info_text_obj is not None:
        self.info_text_obj.remove()
        self.info_text_obj = None
      if self.game_over_text_obj is not None:
        self.game_over_text_obj.remove()
        self.game_over_text_obj = None

  def _game_to_grid(self, game: SnakeGame) -> np.ndarray:
    """Convert game state to numpy grid.
    
    Returns:
        Grid where 0=empty (black), 1=snake (green), 2=food (red)
    """
    grid = np.zeros((game.width, game.height), dtype=int)
    
    # Mark snake positions
    for pt in game.snake:
      if 0 <= pt.x < game.width and 0 <= pt.y < game.height:
        grid[pt.x, pt.y] = 1
    
    # Mark food position
    if 0 <= game.food.x < game.width and 0 <= game.food.y < game.height:
      grid[game.food.x, game.food.y] = 2
    
    return grid

  def _games_to_average_grid(self) -> np.ndarray:
    """Convert multiple games to average grid visualization."""
    snake_grid = np.zeros((self.game.width, self.game.height), dtype=float)
    food_grid = np.zeros((self.game.width, self.game.height), dtype=float)
    
    # Count occurrences at each position
    for game in self.games:
      # Count snake segments
      for i, pt in enumerate(game.snake):
        if 0 <= pt.x < self.game.width and 0 <= pt.y < self.game.height:
          # Weight head more than tail
          weight = 1.0 - (i / max(len(game.snake), 1)) * 0.5
          snake_grid[pt.x, pt.y] += weight
      
      # Count food
      if 0 <= game.food.x < self.game.width and 0 <= game.food.y < self.game.height:
        food_grid[game.food.x, game.food.y] += 1.0
    
    # Normalize
    max_snake = max(snake_grid.max(), 1.0)
    max_food = max(food_grid.max(), 1.0)
    
    # Create combined grid (prioritize food over snake)
    grid = np.zeros((self.game.width, self.game.height), dtype=int)
    
    # Mark food positions (with intensity)
    for x in range(self.game.width):
      for y in range(self.game.height):
        if food_grid[x, y] > 0:
          grid[x, y] = 2
        elif snake_grid[x, y] > 0:
          grid[x, y] = 1
    
    return grid

  def render(self):
    """Render the game state using matplotlib."""
    # Update grid data
    if len(self.games) > 1:
      # Render average of multiple games
      grid = self._games_to_average_grid()
    else:
      # Render single game
      grid = self._game_to_grid(self.game)
    
    # Transpose for correct orientation (matplotlib uses (y, x) convention)
    # We want x on horizontal axis, y on vertical axis
    display_grid = grid.T
    
    # Update image data (much faster than recreating)
    self.im.set_data(display_grid)
    
    # Update score and reward text
    if len(self.games) > 1:
      avg_score = sum(g.score for g in self.games) / len(self.games)
      avg_reward = sum(g.reward for g in self.games) / len(self.games)
      info_text = f"Avg Score: {avg_score:.1f} ({len(self.games)} agents)\nAvg Reward: {avg_reward:.1f}"
      color = 'green' if avg_reward >= 0 else 'red'
    else:
      info_text = f"Score: {self.game.score}\nReward: {self.game.reward}"
      if self.game.step_reward != 0:
        info_text += f" ({'+' if self.game.step_reward > 0 else ''}{self.game.step_reward:.2f})"
      color = 'green' if self.game.reward >= 0 else 'red'
    
    # Update or create info text
    if self.info_text_obj is None:
      self.info_text_obj = self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                                        fontsize=10, verticalalignment='top', color=color,
                                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    else:
      self.info_text_obj.set_text(info_text)
      self.info_text_obj.set_color(color)
    
    # Update game over text
    is_game_over = len(self.games) == 1 and self.game.is_game_over()
    if is_game_over:
      if self.game_over_text_obj is None:
        self.game_over_text_obj = self.ax.text(0.5, 0.5, 'GAME OVER', transform=self.ax.transAxes,
                                              fontsize=24, fontweight='bold', color='red',
                                              ha='center', va='center',
                                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    else:
      if self.game_over_text_obj is not None:
        self.game_over_text_obj.remove()
        self.game_over_text_obj = None
    
    # Use draw_idle for non-blocking updates (much faster than draw())
    # Only flush events if we need to process input
    if self.tick_rate == 0:
      # For maximum speed, only update when necessary
      self.fig.canvas.draw_idle()
    else:
      # For controlled rate, flush events to process input
      self.fig.canvas.draw_idle()
      self.fig.canvas.flush_events()
      time.sleep(1 / self.tick_rate)

  def render_text(self, text: str):
    """Render text message on the plot."""
    self.ax.clear()
    self.ax.text(0.5, 0.5, text, transform=self.ax.transAxes,
                fontsize=16, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    self.ax.set_xticks([])
    self.ax.set_yticks([])
    plt.draw()
    # plt.pause(0.01)

  def _on_key_press(self, event):
    """Handle key press events from matplotlib."""
    if event.key is None:
      return
    
    key = event.key.lower()
    self.key_pressed = key
    
    # Handle quit
    if key == 'q' or key == 'escape':
      self.quit_requested = True
      return
    
    # Handle arrow keys (different backends use different names)
    if key in ['left', 'arrowleft']:
      key = 'left'
    elif key in ['right', 'arrowright']:
      key = 'right'
    elif key in ['up', 'arrowup']:
      key = 'up'
    elif key in ['down', 'arrowdown']:
      key = 'down'
    
    if self.play_mode == 'absolute':
      if key == 'left' and self.game.direction != Vector(1, 0):
        self.game.direction = Vector(-1, 0)
      elif key == 'right' and self.game.direction != Vector(-1, 0):
        self.game.direction = Vector(1, 0)
      elif key == 'up' and self.game.direction != Vector(0, 1):
        self.game.direction = Vector(0, -1)
      elif key == 'down' and self.game.direction != Vector(0, -1):
        self.game.direction = Vector(0, 1)
    elif self.play_mode == 'relative':
      if key == 'left':
        self.game.direction = self.game.get_relative_direction('left')
      elif key == 'right':
        self.game.direction = self.game.get_relative_direction('right')

  def _on_close(self, event):
    """Handle window close event."""
    self.quit_requested = True

  def handle_user_input(self):
    """Handle user input events."""
    # Only flush events if we're not already doing it in render
    # (when tick_rate > 0, render already flushes events)
    # if self.tick_rate == 0:
    self.fig.canvas.flush_events()
    
    # Check for quit
    if self.quit_requested:
      self.close()
      return True
    
    return False

  def close(self):
    """Close the matplotlib window."""
    plt.close(self.fig)
    plt.ioff()


if __name__ == '__main__':
  # TODO: size and tickrate from args, and render mode
  game = SnakeGame(43, 43)
  frontend = SnakeGameMatplotlibFrontend(game, tick_rate=30, play_mode='absolute')

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

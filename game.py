import pygame
import random
import math
from enum import Enum
from collections import namedtuple
from typing import Tuple, Optional
import numpy as np
from config import GameConfig

pygame.init()

class Direction(Enum):
  RIGHT = 1
  LEFT = 2
  UP = 3
  DOWN = 4

Point = namedtuple('Point', 'x, y')

GRID_WIDTH: int = 33
GRID_HEIGHT: int = 33
BLOCK_SIZE: int = 20  # Pixel size for rendering
SPEED: int = 10
MAX_FRAME_ITERATION_MULTIPLIER: int = 10  # max frames = multiplier * snake_length

# Colors (RGB)
WHITE: tuple = (255, 255, 255)
RED: tuple = (200, 0, 0)
BLUE1: tuple = (0, 0, 255)
BLUE2: tuple = (0, 100, 255)
BLACK: tuple = (0, 0, 0)
GRAY: tuple = (100, 100, 100)  # Wall color

# Rewards
REWARD_FOOD: int = 50
REWARD_COLLISION: int = -100
REWARD_DISTANCE_WEIGHT: float = 1.0

FONT_NAME: str = 'arial'
FONT_SIZE: int = 24

WINDOW_WIDTH = GRID_WIDTH * BLOCK_SIZE

def g2p(grid_x: int, grid_y: int) -> Tuple[int, int]:
  return grid_x * BLOCK_SIZE, grid_y * BLOCK_SIZE

def p2g(pixel_x: int, pixel_y: int) -> Tuple[int, int]:
  return pixel_x // BLOCK_SIZE, pixel_y // BLOCK_SIZE

class SnakeGame:
  def __init__(self, config: Optional[GameConfig] = None):
    self.grid_width = GRID_WIDTH
    self.grid_height = GRID_HEIGHT

    # Initialize display
    self.display = pygame.display.set_mode(g2p(self.grid_width, self.grid_height))
    pygame.display.set_caption('Snake')
    self.clock = pygame.time.Clock()
    self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
    self._step_reward = 0.0
    self.reset()

  def reset(self):
    self.direction = random.choice(list(Direction))

    # Start in grid coordinates (center of grid)
    grid_x = random.randint(2, self.grid_width - 4)
    grid_y = random.randint(2, self.grid_height - 4)
    self.head = Point(grid_x, grid_y)
    self.snake = [
      self.head
    ]

    self.score = 0
    self.food = None
    self._place_food()

  def _place_food(self):
    max_attempts = self.grid_width * self.grid_height
    for _ in range(max_attempts):
      x = random.randint(0, self.grid_width - 1)
      y = random.randint(0, self.grid_height - 1)
      candidate = Point(x, y)
      if candidate not in self.snake:
        self.food = candidate
        return
    
    # Fallback: if all cells are occupied (shouldn't happen in normal gameplay)
    self.food = Point(0, 0)

  def is_out_of_bounds(self, pt: Optional[Point] = None):
    if pt is None:
      pt = self.head

    # Check boundary collision (in grid coordinates)
    if pt.x < 0 or pt.x >= self.grid_width or pt.y < 0 or pt.y >= self.grid_height:
      return True
    
  def is_collision(self, pt: Optional[Point] = None) -> bool:
    if pt is None:
      pt = self.head
    
    # Check self-collision
    return pt in self.snake[1:]


  def is_game_over(self, pt: Optional[Point] = None) -> bool:
    return self.is_out_of_bounds(pt) or self.is_collision(pt)

  def _get_next_head_position(self) -> Point:
    return self.get_next_position(self.head, self.direction)

  def get_next_position(self, point: Point, direction: Direction) -> Point:
    direction_deltas = {
      Direction.RIGHT: (1, 0),
      Direction.LEFT: (-1, 0),
      Direction.DOWN: (0, 1),
      Direction.UP: (0, -1),
    }
    dx, dy = direction_deltas[direction]
    return Point(point.x + dx, point.y + dy)

  def get_relative_direction(self, relative: str) -> Direction:
    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    idx = clock_wise.index(self.direction)
    
    if relative == 'straight':
      return clock_wise[idx]
    elif relative == 'right':
      next_idx = (idx + 1) % 4
      return clock_wise[next_idx]
    elif relative == 'left':
      next_idx = (idx - 1) % 4
      return clock_wise[next_idx]
    else:
      raise ValueError(f"Invalid relative direction: {relative}. Must be 'straight', 'right', or 'left'")

  def _move(self, direction: Direction):
    self.direction = direction
    self.head = self._get_next_head_position()

  def _handle_events(self):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()

  def _transform_point(self, pt: Point) -> Tuple[int, int]:
    """Transform a grid point to screen coordinates with rotation.
    The snake head is always at the center, and the snake always points up.
    """
    # Calculate offset from snake head
    dx = pt.x - self.head.x
    dy = pt.y - self.head.y
    
    # Map direction to rotation angle (in radians)
    # UP = 0°, RIGHT = -90°, DOWN = 180°, LEFT = 90°
    direction_angles = {
      Direction.UP: 0,
      Direction.RIGHT: -math.pi / 2,
      Direction.DOWN: math.pi,
      Direction.LEFT: math.pi / 2,
    }
    angle = direction_angles[self.direction]
    
    # Rotate the offset vector
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated_dx = dx * cos_a - dy * sin_a
    rotated_dy = dx * sin_a + dy * cos_a
    
    # Center of screen
    center_x = WINDOW_WIDTH // 2
    center_y = (self.grid_height * BLOCK_SIZE) // 2
    
    # Translate to screen coordinates
    screen_x = center_x + rotated_dx * BLOCK_SIZE
    screen_y = center_y + rotated_dy * BLOCK_SIZE
    
    return int(screen_x), int(screen_y)

  def _update_ui(self):
    self.display.fill(BLACK)

    # Draw walls (with rotation)
    # Top wall
    for x in range(self.grid_width):
      wall_x, wall_y = self._transform_point(Point(x, -1))
      wall_rect = pygame.Rect(wall_x, wall_y, BLOCK_SIZE, BLOCK_SIZE)
      pygame.draw.rect(self.display, GRAY, wall_rect)
    
    # Bottom wall
    for x in range(self.grid_width):
      wall_x, wall_y = self._transform_point(Point(x, self.grid_height))
      wall_rect = pygame.Rect(wall_x, wall_y, BLOCK_SIZE, BLOCK_SIZE)
      pygame.draw.rect(self.display, GRAY, wall_rect)
    
    # Left wall
    for y in range(self.grid_height):
      wall_x, wall_y = self._transform_point(Point(-1, y))
      wall_rect = pygame.Rect(wall_x, wall_y, BLOCK_SIZE, BLOCK_SIZE)
      pygame.draw.rect(self.display, GRAY, wall_rect)
    
    # Right wall
    for y in range(self.grid_height):
      wall_x, wall_y = self._transform_point(Point(self.grid_width, y))
      wall_rect = pygame.Rect(wall_x, wall_y, BLOCK_SIZE, BLOCK_SIZE)
      pygame.draw.rect(self.display, GRAY, wall_rect)

    # Draw snake (with rotation - head always centered, pointing up)
    for pt in self.snake:
      pixel_x, pixel_y = self._transform_point(pt)
      block_rect = pygame.Rect(pixel_x, pixel_y, BLOCK_SIZE, BLOCK_SIZE)
      pygame.draw.rect(self.display, BLUE1, block_rect)
      inner_rect = pygame.Rect(pixel_x + 4, pixel_y + 4, 12, 12)
      pygame.draw.rect(self.display, BLUE2, inner_rect)

    # Draw food (with rotation)
    food_x, food_y = self._transform_point(self.food)
    food_rect = pygame.Rect(food_x, food_y, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(self.display, RED, food_rect)

    # Draw score (not rotated, stays in top-left)
    text = self.font.render(f"Score: {self.score}", True, WHITE)
    self.display.blit(text, [0, 0])
    
    # Draw reward if in AI mode (not rotated, stays in top-left)
    if hasattr(self, 'frame_iteration'):
      reward_text = self.font.render(f"Reward: {self._step_reward:.2f}", True, WHITE)
      self.display.blit(reward_text, [0, 25])
    
    pygame.display.flip()

  def _prepare_move(self):
    pass

  def _get_next_direction(self) -> Direction:
    return self.direction

  def _calculate_game_over(self) -> bool:
    return self.is_game_over()

  def _calculate_reward(self) -> float:
    return 0.0

  def _post_move_pre_food(self) -> float:
    return 0.0

  def _execute_step(self) -> Tuple[bool, int]:
    self._handle_events()
    
    # Prepare move (e.g., calculate distance before move)
    self._prepare_move()
    
    # Get next direction and move
    next_direction = self._get_next_direction()
    self._move(next_direction)
    self.snake.insert(0, self.head)
    
    # Calculate reward after move (before food check)
    base_reward = 0#self._post_move_pre_food()
    
    # Check for game over
    game_over = self._calculate_game_over()
    
    if game_over:
      # Store base reward even on game over (subclasses may override collision reward)
      self._step_reward = base_reward
      self._update_ui()
      self.clock.tick(SPEED)
      return game_over, self.score
    
    # Handle food consumption
    food_reward = 0.0
    if self.head == self.food:
      self.score += 1
      # food_reward = REWARD_FOOD if hasattr(self.config, 'REWARD_FOOD') else 0.0
      self._place_food()
    else:
      self.snake.pop()
    
    # Store total reward for subclasses that need it
    self._step_reward = base_reward + food_reward
    
    # Update UI and clock
    self._update_ui()
    self.clock.tick(SPEED)
    return game_over, self.score


class SnakeGameAI(SnakeGame):

  def __init__(self, config: Optional[GameConfig] = None):
    super().__init__(config)
    self.frame_iteration = 0
    self._action = None
    self._distance_before = 0.0

  def reset(self):
    super().reset()
    self.frame_iteration = 0

  def play_step(self, action: np.ndarray):
    self.frame_iteration += 1
    self._action = action
    
    game_over, score = self._execute_step()
    
    # Get reward from step execution (includes food reward if consumed)
    reward = getattr(self, '_step_reward', 0.0)
    
    if game_over:
      reward = REWARD_COLLISION
    elif self._action != [1, 0, 0]:
      reward -= 2
    
    return reward, game_over, score

  def _prepare_move(self):
    self._distance_before = self._calculate_distance_to_food()

  def _get_next_direction(self) -> Direction:
    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    idx = clock_wise.index(self.direction)

    if np.array_equal(self._action, [1, 0, 0]):
      new_dir = clock_wise[idx]  # no change
    elif np.array_equal(self._action, [0, 1, 0]):
      next_idx = (idx + 1) % 4
      new_dir = clock_wise[next_idx]  # right turn
    else:  # [0, 0, 1]
      next_idx = (idx - 1) % 4
      new_dir = clock_wise[next_idx]  # left turn

    return new_dir

  def _calculate_game_over(self) -> bool:
    max_frames = MAX_FRAME_ITERATION_MULTIPLIER * len(self.snake)**2
    return self.is_game_over() or self.frame_iteration > max_frames

  def _post_move_pre_food(self) -> float:
    """Calculate distance-based reward after move.
    
    Returns deterministic reward based on distance change:
    - Positive reward for moving closer to food
    - Negative reward for moving away from food
    - Small negative reward per step to encourage efficiency
    """
    distance_after = self._calculate_distance_to_food()
    distance_before = self._distance_before
    
    # Calculate distance change (negative means closer)
    distance_change = distance_after - distance_before
    
    # Continuous reward based on distance improvement
    # Normalize by grid size to keep rewards in reasonable range
    max_distance = math.sqrt(
      self.grid_width ** 2 + self.grid_height ** 2
    )
    distance_reward = -distance_change / max_distance * REWARD_DISTANCE_WEIGHT * 10
    
    # Small negative reward per step to encourage efficiency
    step_penalty = -0.1
    
    return distance_reward + step_penalty

  def _calculate_distance_to_food(self) -> float:
    return math.sqrt(
      (self.food.x - self.head.x) ** 2 + 
      (self.food.y - self.head.y) ** 2
    )


class SnakeGameHuman(SnakeGame):
  def __init__(self, config=None):
    super().__init__(config)
    self._last_left_pressed = False
    self._last_right_pressed = False

  def play_step(self):
    self._handle_events()
    return self._execute_step()

  def _handle_events(self):
    # Handle quit events
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()

    # Use key state checking to catch fast key presses
    # This checks the current state of keys every frame, so we don't miss fast presses
    keys = pygame.key.get_pressed()
    
    # Check for key presses (only process if key is newly pressed, not held)
    left_pressed = keys[pygame.K_LEFT]
    right_pressed = keys[pygame.K_RIGHT]
    
    if left_pressed and not self._last_left_pressed:
      # Turn left relative to current direction
      self.direction = self.get_relative_direction('left')
    elif right_pressed and not self._last_right_pressed:
      # Turn right relative to current direction
      self.direction = self.get_relative_direction('right')
    
    # Update last keys state for next frame
    self._last_left_pressed = left_pressed
    self._last_right_pressed = right_pressed

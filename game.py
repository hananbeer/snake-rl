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

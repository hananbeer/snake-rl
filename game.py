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
  def __init__(self, width: int, height: int, seed: Optional[int] = None, allow_teleport: bool = False):
    self.width = width
    self.height = height
    self.seed = seed
    self.rng = random.Random(seed) if seed is not None else random
    self.allow_teleport = allow_teleport
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


  def teleport(self, pt: Point) -> Point:
    return Point((pt.x + self.width) % self.width, (pt.y + self.height) % self.height)


  def advance_head(self) -> Point:
    next_head = self.get_next_position(self.head, self.direction)
    if self.allow_teleport:
      next_head = self.teleport(next_head)

    self.snake.insert(0, next_head)
    return next_head


  def pop_tail(self):
    self.snake.pop()


  def step(self) -> Tuple[bool, int]:
    next_head = self.advance_head()

    ate_food = (next_head == self.food)
    if ate_food:
      self.food = self.make_food()
    else:
      self.pop_tail()

    is_game_over = self.is_game_over()
    return ate_food, is_game_over

def fixed_view(arr: np.ndarray, x: int, y: int, view_width: int, view_height: int, default_value = np.nan) -> np.ndarray:
  """
  @param arr: the array to get the view of
  @param x: the x coordinate of the top left corner of the view
  @param y: the y coordinate of the top left corner of the view
  @param view_width: the width of the view
  @param view_height: the height of the view
  @param default_value: the value to fill the view with if the view is out of bounds
  @return: a fixed size view of an array
  """
  arr_height, arr_width = arr.shape
  x1 = max(x, 0)
  y1 = max(y, 0)
  x2 = min(x + view_width, arr_width)
  y2 = min(y + view_height, arr_height)

  target_x = x1 - x
  target_y = y1 - y
  extracted_width = x2 - x1
  extracted_height = y2 - y1

  view = np.full((view_height, view_width), default_value)
  view[target_y:target_y + extracted_height, target_x:target_x + extracted_width] = arr[y1:y2, x1:x2]

  return view

def wrapped_view(arr: np.ndarray, x: int, y: int, view_width: int, view_height: int) -> np.ndarray:
  """
  @param arr: the array to get the view of
  @param x: the x coordinate of the top left corner of the view
  @param y: the y coordinate of the top left corner of the view
  @param view_width: the width of the view
  @param view_height: the height of the view
  @return: a fixed size view of an array
  """
  arr_height, arr_width = arr.shape
  half_height = view_height // 2
  half_width = view_width // 2
  view = np.zeros((view_height, view_width), dtype=arr.dtype)
  for dy in range(view_height):
    for dx in range(view_width):
      arr_y = (y + dy - half_height) % arr_height
      arr_x = (x + dx - half_width) % arr_width
      view[dy, dx] = arr[arr_y, arr_x]

  return view


class SnakeGameAI(SnakeGame):
  def world_view(self) -> np.ndarray:
    grid = np.zeros((self.width, self.height), dtype=int)
    for pt in self.snake[1:]:
      grid[pt.y, pt.x] = 1

    grid[self.head.y, self.head.x] = 2
    grid[self.food.y, self.food.x] = 3
    return grid

  def local_view(self, block_size: int = 5) -> np.ndarray:
    world = self.world_view()
    block_half = block_size // 2
    local = fixed_view(world, self.head.x - block_half, self.head.y - block_half, block_size, block_size)

    # Determine the number of CCW 90-degree rotations to align self.direction to up (0, -1)
    rotation_map = {
        (0, -1): 0,   # Up: no rotation
        (-1, 0): 1,   # Left: rotate 90° CCW
        (0, 1): 2,    # Down: rotate 180°
        (1, 0): 3,    # Right: rotate 270° CCW
    }
    dir_tuple = (self.direction.x, self.direction.y)
    k = rotation_map.get(dir_tuple, 0)
    local = np.rot90(local, k)

    return local

  def wrapped_view(self, x: int, y: int, block_size: int = 5) -> np.ndarray:
    world = self.world_view()
    return wrapped_view(world, x, y, block_size, block_size)

  def step(self):
    self.step_reward = 0

    prev_head = self.head
    ate_food, is_game_over = super().step()

    if ate_food:
      self.score += 1
      self.step_reward += 50
    else:
      # dist = math.sqrt((next_head.x - self.food.x) ** 2 + (next_head.y - self.food.y) ** 2)
      # self.step_reward -= dist - 5
      if (self.head.x == self.food.x or self.head.y == self.food.y) and not (prev_head.x == self.food.x or prev_head.y == self.food.y):
        self.step_reward += 1
      elif not (prev_head.x == self.food.x or prev_head.y == self.food.y) and (self.head.x == self.food.x or self.head.y == self.food.y):
        self.step_reward -= 5

    return ate_food, is_game_over

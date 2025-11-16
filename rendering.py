import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import gridspec, colors
from typing import Tuple

import game

class MPLRenderer:
  def __init__(
    self,
    figsize: Tuple[int, int] = (20, 8),
    rows: int = 2,
    cols: int = 4,
  ):
    self.fig = plt.figure(figsize=figsize)
    self.rows = rows
    self.cols = cols
    self.axes = {}
    self.ax = None

  def subplot(self, name, index=None):
    if name in self.axes:
      return self.axes[name]

    if index is None:
      try:
        index = list(self.axes.keys()).index(name) + 1
      except ValueError:
        index = len(self.axes) + 1

    ax = self.fig.add_subplot(self.rows, self.cols, index)
    self.axes[name] = ax
    return ax

  def render_world_view(self, game: game.SnakeGameAI, ax = None):
    if ax is None:
      ax = self.subplot('World View')

    ax.clear()
    world_grid = game.world_view()
    ax.imshow(world_grid, vmin=0, vmax=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('World View')

  def render_local_view(self, game: game.SnakeGameAI, block_size = 5, block_pad = 0, ax = None):
    if ax is None:
      ax = self.subplot('Local View')

    ax.clear()
    local_grid = game.local_view(block_size=block_size)
    local_grid_padded = np.full((block_size + 2 * block_pad, block_size + 2 * block_pad), np.nan)
    local_grid_padded[block_pad:block_pad + block_size, block_pad:block_pad + block_size] = local_grid

    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='black') # Set 'bad' values (masked as nan) to black
    ax.imshow(local_grid_padded, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Local View')

  def render_wrapped_view(self, game: game.SnakeGameAI, block_size = 5, ax = None):
    if ax is None:
      ax = self.subplot('Wrapped View')

    ax.clear()
    wrapped_grid = game.wrapped_view(x=game.head.x, y=game.head.y, block_size=block_size)
    ax.imshow(wrapped_grid, vmin=0, vmax=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Wrapped View')

  def show(self, *args, **kwargs):
    plt.show(*args, **kwargs)


if __name__ == '__main__':
  g = game.SnakeGameAI(20, 20)#, seed=42)#, allow_teleport=True)
  g.reset()
  # direction = game.Vector(0, 1)
  # g.direction = direction

  renderer = MPLRenderer(figsize=(18, 5), rows=2, cols=2)

  for i in range(100):
    renderer.render_world_view(g)
    renderer.render_local_view(g, block_size=9, block_pad=2)
    renderer.render_wrapped_view(g, block_size=21)
    renderer.show(block=False)
    # plt.pause(0.1)

    # renderer.fig.canvas.draw()
    renderer.fig.canvas.flush_events()
    ate_food, game_over = g.step()
    if ate_food:
      print(f"Ate food at step {i}")

    if game_over:
      print(f"Game over at step {i}")
      g.reset()
      # g.direction = direction

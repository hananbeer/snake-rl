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
      ax = self.axes[name]
      ax.clear()
      ax.set_title(name)
      return ax

    if index is None:
      index = len(self.axes) + 1
      print(index)

    ax = self.fig.add_subplot(self.rows, self.cols, index)
    self.axes[name] = ax
    ax.set_title(name)
    return ax

  def render_world_view(self, game: game.SnakeGameAI, rotate: bool = False):
    ax = self.subplot('World View' + (' (rotated)' if rotate else ''))
    world_grid = game.world_view(rotate=rotate)
    ax.imshow(world_grid, vmin=0, vmax=3)
    ax.set_xticks([])
    ax.set_yticks([])

  def render_local_view(self, game: game.SnakeGameAI, block_size = 5, block_pad = 0, rotate: bool = False):
    ax = self.subplot('Local View' + (' (rotated)' if rotate else ''))
    local_grid = game.local_view(block_size=block_size, rotate=rotate)
    local_grid_padded = np.full((block_size + 2 * block_pad, block_size + 2 * block_pad), -np.inf)
    local_grid_padded[block_pad:block_pad + block_size, block_pad:block_pad + block_size] = local_grid

    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='black') # Set 'bad' values (masked as nan) to black
    ax.imshow(local_grid_padded, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks([])
    ax.set_yticks([])

  def render_wrapped_view(self, game: game.SnakeGameAI, block_size = 5, rotate: bool = False):
    ax = self.subplot('Wrapped View' + (' (rotated)' if rotate else ''))
    wrapped_grid = game.wrapped_view(x=game.head.x, y=game.head.y, block_size=block_size, rotate=rotate)
    ax.imshow(wrapped_grid, vmin=0, vmax=3)
    ax.set_xticks([])
    ax.set_yticks([])

  def render_smell_view(self, game: game.SnakeGameAI, block_size = 5, rotate: bool = False):
    ax = self.subplot('Smell View' + (' (rotated)' if rotate else ''))
    smell_grid = game.smell_view(block_size=block_size, rotate=rotate)
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='black')
    ax.imshow(smell_grid, cmap=cmap, vmin=0, vmax=1/block_size**0.15)
    ax.set_xticks([])
    ax.set_yticks([])

  def show(self, *args, **kwargs):
    plt.show(*args, **kwargs)


if __name__ == '__main__':
  g = game.SnakeGameAI(20, 20)#, seed=42, allow_teleport=True)
  g.reset()
  # direction = game.Vector(1, 0)
  # g.direction = direction

  renderer = MPLRenderer(figsize=(10, 8), rows=2, cols=4)
  renderer.fig.subplots_adjust(wspace=0.1, hspace=.14)

  steps = 0
  for i in range(500):
    for rotate in [False, True]:
      renderer.render_world_view(g, rotate=rotate)
      renderer.render_local_view(g, block_size=15, block_pad=0, rotate=rotate)
      renderer.render_wrapped_view(g, block_size=21, rotate=rotate)
      renderer.render_smell_view(g, block_size=25, rotate=rotate)

    renderer.show(block=False)
    # plt.pause(0.25)

    # renderer.fig.canvas.draw()
    renderer.fig.canvas.flush_events()
    ate_food, game_over = g.step()
    steps += 1
    if ate_food:
      print(f"Ate food at step {steps}")

    if game_over:
      print(f"Game over at step {steps}")
      g.reset()
      steps = 0
      # g.direction = direction

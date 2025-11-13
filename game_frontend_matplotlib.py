import pygame
import numpy as np
import time
from typing import Tuple, Optional, List
from game import SnakeGame, Vector
import matplotlib.pyplot as plt

SPEED = 10

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 200, 0)
GREEN2 = (0, 100, 0)
GRAY = (100, 100, 100)

class SnakeGameMatplotlibFrontend:
  def __init__(self, game: SnakeGame, tick_rate: float = 0, games: Optional[List[SnakeGame]] = None, play_mode: str = 'absolute'):
    self.game = game
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
    self.ax.set_title('Snake Game', fontsize=16, fontweight='bold')
    self.ax.set_aspect('equal')
    self.ax.set_xticks([])
    self.ax.set_yticks([])
    plt.tight_layout()
    
    # Create initial empty grid for the image (will be updated with set_data)
    initial_grid = np.zeros((game.height, game.width), dtype=int)
    self.im = self.ax.imshow(initial_grid, vmin=0, vmax=2, 
                            interpolation='nearest', origin='upper', 
                            animated=True,
                            cmap='viridis')
    
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
    # if 0 <= game.food.x < game.width and 0 <= game.food.y < game.height:
    grid[game.food.x, game.food.y] = 2
    
    return grid

  def render(self):
    """Render the game state using matplotlib."""
    # Update grid data
    grid = self._game_to_grid(self.game)
    
    # Transpose for correct orientation (matplotlib uses (y, x) convention)
    # We want x on horizontal axis, y on vertical axis
    display_grid = grid.T
    
    # Update image data (much faster than recreating)
    self.im.set_data(display_grid)
    
    # Update score and reward text
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
    if self.game.is_game_over():
      if self.game_over_text_obj is None:
        self.game_over_text_obj = self.ax.text(0.5, 0.5, 'GAME OVER', transform=self.ax.transAxes,
                                              fontsize=24, fontweight='bold', color='red',
                                              ha='center', va='center',
                                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
      else:
        self.game_over_text_obj.set_visible(True)
    else:
      if self.game_over_text_obj is not None:
        self.game_over_text_obj.set_visible(False)
    
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
    if event.key is None or self.key_pressed:
      return
    
    key = event.key.lower()
    
    # Handle quit
    if key == 'q' or key == 'escape':
      self.quit_requested = True
      return
    
    self.key_pressed = key.replace('arrow', '')

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
    
    key = self.key_pressed
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

    self.key_pressed = None

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

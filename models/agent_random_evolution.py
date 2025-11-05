import time
import torch
import random
import math
import numpy as np
from collections import deque

import sys
import os

from game import SnakeGame, SnakeGameAI, SnakeGameFrontend, SnakeGameMatplotlibFrontend, Point, Vector
from .model_simple_layernorm import ModelSimpleLayerNorm
# from .qtrainer_simple import QTrainerSimple
from .qtrainer import QTrainer
from .model_visualizer import ModelVisualizer
from .signal_visualizer import SignalVisualizer

import signals

import helper

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
TARGET_UPDATE_FREQUENCY = 15  # Update target network every N games
EPSILON_START = 0.6
EPSILON_END = 0.001
EPSILON_DECAY = 0.995

class Agent:
  def __init__(self, model=None):
    self.n_games = 0
    if model is None:
      self.model = ModelSimpleLayerNorm(7, 512, 3)
    else:
      self.model = model


  def get_state(self, game):
    head = game.snake[0]
    point_l = Point(head.x - 1, head.y)
    point_r = Point(head.x + 1, head.y)
    point_u = Point(head.x, head.y - 1)
    point_d = Point(head.x, head.y + 1)
    
    dir_l = game.direction == Vector(-1, 0)
    dir_r = game.direction == Vector(1, 0)
    dir_u = game.direction == Vector(0, -1)
    dir_d = game.direction == Vector(0, 1)

    state = [
      *signals.get_danger_signals(game),
      
      # Move direction
      # *signals.get_direction_bitmap_signals(game), # indeed it works without it better?

      # game.direction.x,
      # game.direction.y,
      
      # Food location 
      game.food.x < game.head.x,  # food left
      game.food.x > game.head.x,  # food right
      game.food.y < game.head.y,  # food up
      game.food.y > game.head.y,  # food down


      # Food location v2
      # (game.food.x < game.head.x) == (game.direction.x < 0),  # food left
      # # (game.food.x > game.head.x) == (game.direction.x > 0),  # food right
      # (game.food.y < game.head.y) == (game.direction.y < 0),  # food up
      # # (game.food.y > game.head.y) == (game.direction.y > 0),  # food down

      # Food distance
      # math.sqrt((game.food.x - game.head.x) ** 2 + (game.food.y - game.head.y) ** 2) # distance from food
    ]

    return np.array(state, dtype=int)

  def predict(self, state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
      prediction = self.model(state_tensor)
    return prediction.squeeze(0).numpy()

  def clone(self):
    """Create a copy of this agent with the same model weights."""
    new_agent = Agent()
    new_agent.model.load_state_dict(self.model.state_dict())
    return new_agent

  def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
    """Apply random mutations to the model weights."""
    with torch.no_grad():
      for param in self.model.parameters():
        if random.random() < mutation_rate:
          noise = torch.randn_like(param) * mutation_strength
          param.add_(noise)


def evaluate_agent(agent, game, max_steps, index: int, frontend=None):
  """Evaluate an agent by running a game and returning the final score."""
  game.reset()
  steps = 0
  
  while True:
    state = agent.get_state(game)
    pred = agent.predict(state)
    
    idx = np.argmax(pred)
    if idx == 1:
      game.direction = game.get_relative_direction('left')
    elif idx == 2:
      game.direction = game.get_relative_direction('right')
    
    reward, done, score = game.step()
    
    if frontend and index % 1000 == 500:
      frontend.render()
    
      print(f"Evaluating agent {index}", end='\r')
      frontend.render_text(f"Evaluating agent {index}")

    steps += 1
    if done or steps > max_steps * len(game.snake):
      return score / (max_steps / steps)


def evaluate_top_agents(agents, games, max_steps, frontend=None):
  """Evaluate top agents simultaneously and render their average."""
  for game in games:
    game.reset()
  
  steps = 0
  all_done = False
  
  while not all_done:
    all_done = True
    for agent, game in zip(agents, games):
      if game.is_game_over():
        continue
      
      all_done = False
      state = agent.get_state(game)
      pred = agent.predict(state)
      
      idx = np.argmax(pred)
      if idx == 1:
        game.direction = game.get_relative_direction('left')
      elif idx == 2:
        game.direction = game.get_relative_direction('right')
      
      game.step()
    
    if frontend:
      frontend.render()
    
    steps += 1
    if steps > max_steps * max((len(g.snake) for g in games), default=1):
      all_done = True


def train(render: bool, population_size=150, num_generations=100, elite_size=15, mutation_rate=0.1, mutation_strength=0.1, visualize_model: bool = True):
  """Train using genetic algorithm with population-based evolution."""
  plot_scores = []
  plot_mean_scores = []
  record = 0
  
  # Initialize population with random agents
  population = [Agent() for _ in range(population_size)]
  game = SnakeGame(33, 33)
  
  frontend = None
  if render:
    frontend = SnakeGameMatplotlibFrontend(game, tick_rate=0)
  
  # Initialize model visualizer
  visualizer = None
  # if visualize_model:
  #   try:
  #     visualizer = ModelVisualizer(population[0].model, update_interval=0.5)
  #   except Exception as e:
  #     print(f"Warning: Could not initialize model visualizer: {e}")
  #     visualize_model = False
  
  # Initialize signal visualizer
  signal_visualizer = None
  try:
    signal_visualizer = SignalVisualizer(update_interval=0.1, max_history=500)
    # Register signals
    signal_visualizer.register_vector_signal("danger_signals")
    signal_visualizer.register_vector_signal("direction_signals")
    signal_visualizer.register_scalar_signal("score")
    signal_visualizer.register_scalar_signal("reward")
    # Register grid signal (will be used with SnakeGameAI during visualization)
    signal_visualizer.register_grid_signal("world_view")
  except Exception as e:
    print(f"Warning: Could not initialize signal visualizer: {e}")
    signal_visualizer = None
  
  max_steps = 100
  
  for generation in range(num_generations):
    # Evaluate fitness (score) for each agent in the population
    fitness_scores = []
    for i, agent in enumerate(population):
      print(f"Evaluating agent {i}/{len(population)}", end='\r')
      # frontend.render_text(f"Evaluating agent {i}")
      score = evaluate_agent(agent, game, max_steps, i, frontend=frontend)

      fitness_scores.append((score, agent))
      
      # Update visualizer periodically during evaluation (every 10 agents)
      # if visualize_model and visualizer and i % 10 == 0:
      #   visualizer.model = agent.model
      #   visualizer.update()
    
    # Sort by fitness (score) - higher is better
    fitness_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Get best score and mean score
    best_score = fitness_scores[0][0]
    mean_score = sum(score for score, _ in fitness_scores) / len(fitness_scores)
    best_agent = fitness_scores[0][1]
    
    # Update model visualizer with best agent
    # if visualize_model and visualizer:
    #   visualizer.model = best_agent.model
    #   visualizer.update(force=True)
    
    # Render the average of top 10 agents if rendering is enabled
    if render and frontend:
      top_10_agents = [agent for _, agent in fitness_scores[:10]]
      # Use same seed for all games so they start from the same state
      render_seed = game.seed if hasattr(game, 'seed') and game.seed is not None else 42
      top_10_games = [SnakeGame(game.width, game.height, seed=render_seed) for _ in range(10)]
      frontend.games = top_10_games
      evaluate_top_agents(top_10_agents, top_10_games, max_steps, frontend=frontend)
      frontend.games = [game]  # Reset to single game
      
      # Update visualizer during rendering
      # if visualize_model and visualizer:
      #   visualizer.update()

      # VISUAL HERE:
      # Visualize signals for the best agent
      if signal_visualizer and False:
        # Use SnakeGameAI to enable grid visualization
        best_game = SnakeGameAI(game.width, game.height, seed=render_seed)
        best_game.reset()
        steps = 0
        max_visualization_steps = max_steps * 10  # Show more steps for visualization
        
        while steps < max_visualization_steps and not best_game.is_game_over():
          # Get signals
          danger_sigs = signals.get_danger_signals(best_game)
          # direction_sigs = signals.get_direction_bitmap_signals(best_game)
          
          # Update signal visualizer
          signal_visualizer.update_vector("danger_signals", danger_sigs)
          # signal_visualizer.update_vector("direction_signals", direction_sigs)
          # signal_visualizer.update_scalar("score", best_game.score)
          # signal_visualizer.update_scalar("reward", best_game.reward)
          
          # Update grid signal (world_view is available in SnakeGameAI)
          world_grid = best_game.world_view()
          signal_visualizer.update_grid("world_view", world_grid)
          
          # Get agent action
          state = best_agent.get_state(best_game)
          pred = best_agent.predict(state)
          idx = np.argmax(pred)
          if idx == 1:
            best_game.direction = best_game.get_relative_direction('left')
          elif idx == 2:
            best_game.direction = best_game.get_relative_direction('right')
          
          # Step game
          best_game.step()
          
          # Update visualization
          signal_visualizer.update(force=True)
          
          steps += 1
          
          # Break early if game over
          if best_game.is_game_over():
            break
    
    if best_score > record:
      record = best_score
    
    plot_scores.append(best_score)
    plot_mean_scores.append(mean_score)
    
    print(f"Generation {generation + 1}/{num_generations} - Best: {best_score}, Mean: {mean_score:.2f}, Record: {record}")
    
    # Select elite agents (top performers)
    elite = [agent.clone() for _, agent in fitness_scores[:elite_size]]
    
    # Create new generation
    new_population = []
    
    # Keep elite agents
    new_population.extend(elite)
    
    # Fill rest of population with mutations of elite agents, timing the operation
    start_time = time.time()
    while len(new_population) < population_size:
      # Select a random elite agent
      parent = random.choice(elite)
      # Create mutated offspring
      offspring = parent.clone()
      offspring.mutate(mutation_rate, mutation_strength)
      new_population.append(offspring)
    elapsed_time = time.time() - start_time
    print(f"Time to fill population: {elapsed_time:.4f} seconds")
    population = new_population
    
    # Update visualizer periodically
    if visualize_model and visualizer:
      visualizer.update()
  
  # Cleanup
  if visualize_model and visualizer:
    print("\nModel visualization window will remain open.")
    print("Close the matplotlib window manually or it will close when the program exits.")
  
  if signal_visualizer:
    print("\nSignal visualization window will remain open.")
    print("Close the matplotlib window manually or it will close when the program exits.")
  
  print(f"\nTraining complete! Final record: {record}")
  return plot_scores, plot_mean_scores
      


if __name__ == '__main__':
  os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'
  train(render = True)
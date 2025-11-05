import torch
import random
import math
import numpy as np
from collections import deque

import sys
import os

from game import SnakeGame, SnakeGameFrontend, Point, Vector
from .model_simple_layernorm import ModelSimpleLayerNorm
# from .qtrainer_simple import QTrainerSimple
from .qtrainer import QTrainer

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
  def __init__(self):
    self.n_games = 0
    self.epsilon = EPSILON_START # randomness
    self.gamma = 0.9 # discount rate
    self.memory = deque(maxlen=MAX_MEMORY) # popleft()
    self.model = ModelSimpleLayerNorm(11, 512, 3)
    self.target_model = ModelSimpleLayerNorm(11, 512, 3)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()
    self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)


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
      *signals.get_direction_bitmap_signals(game),

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

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

  def train_long_memory(self):
    if len(self.memory) > BATCH_SIZE:
      mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
    else:
      mini_sample = self.memory

    states, actions, rewards, next_states, dones = zip(*mini_sample)
    self.trainer.train_step(states, actions, rewards, next_states, dones)
    #for state, action, reward, nexrt_state, done in mini_sample:
    #    self.trainer.train_step(state, action, reward, next_state, done)

  def train_short_memory(self, state, action, reward, next_state, done):
      self.trainer.train_step(state, action, reward, next_state, done)

  def get_action(self, state):
    # Exponential epsilon decay for exploration/exploitation tradeoff
    final_move = [0,0,0]
    r = random.random()
    if r < self.epsilon:
      move = random.randint(0, 2)
      final_move[move] = 1
      # print('random move', r)#, end='\r')
    else:
      state0 = torch.tensor(state, dtype=torch.float)
      self.model.eval()
      with torch.no_grad():
        prediction = self.model(state0)
      self.model.train()
      move = torch.argmax(prediction).item()
      final_move[move] = 1
      # print('model move', r)#, end='\r')

    return final_move

  def update_epsilon(self):
    # Exponential decay: epsilon = max(epsilon_end, epsilon * decay_rate)
    self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    print('epsilon', self.epsilon)

  def update_target_network(self):
    # Copy weights from main network to target network
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()


def train(render: bool):
  plot_scores = []
  plot_mean_scores = []
  total_score = 0
  record = 0
  agent = Agent()
  game = SnakeGame(11, 11)

  frontend = None
  if render:
    frontend = SnakeGameFrontend(game, tick_rate = 0)

  max_steps = 100

  steps = 0

  while True:
    # get old state
    state_old = agent.get_state(game)

    # get action logits [forward, right, left]
    final_move = agent.get_action(state_old)

    # take action
    idx = np.argmax(final_move)
    if (idx == 1):
      game.direction = game.get_relative_direction('left')
    elif (idx == 2):
      game.direction = game.get_relative_direction('right')

    # perform move and get new state
    reward, done, score = game.step()

    # if idx != 0:
    #   reward -= 1

    if frontend:
      frontend.render()

    state_new = agent.get_state(game)

    # train short memory
    agent.train_short_memory(state_old, final_move, reward, state_new, done)

    # remember
    agent.remember(state_old, final_move, reward, state_new, done)

    steps += 1
    if steps > max_steps * len(game.snake):
      done = True

    if done:
      steps = 0
      # train long memory, plot result
      game.reset()
      agent.n_games += 1
      agent.train_long_memory()
      
      # Update epsilon with exponential decay
      agent.update_epsilon()
      
      # Update target network periodically
      if agent.n_games % TARGET_UPDATE_FREQUENCY == 0:
        agent.update_target_network()
      
      # Update learning rate scheduler
      agent.trainer.update_scheduler()

      # if score > record:
      #   record = score
      #   agent.model.save()

      print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Epsilon:', f'{agent.epsilon:.3f}')

      plot_scores.append(score)
      total_score += score
      mean_score = total_score / agent.n_games
      plot_mean_scores.append(mean_score)
      helper.plot(plot_scores, plot_mean_scores)

      plot_scores = plot_scores[-200:]
      plot_mean_scores = plot_mean_scores[-200:]

      next_size = 13 + int(mean_score * 4 + 0.8)
      if next_size != game.width:
        game = SnakeGame(next_size, next_size)
        if frontend:
          frontend.resize(game)


if __name__ == '__main__':
  train(render = True)
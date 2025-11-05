import sys
import argparse
import importlib

# to set the window pos
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = '1280,-400'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='agent_random_evolution')
args = parser.parse_args()

model = importlib.import_module(f'models.{args.model}')
model.train(render=True)

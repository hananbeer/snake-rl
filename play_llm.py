import os
import time
from typing import Optional

import requests

from game import SnakeGame, Vector, Point
from game_frontend_matplotlib import SnakeGameMatplotlibFrontend

import dotenv
dotenv.load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DIRECTION_VECTORS = {
  "up": Vector(0, -1),
  "down": Vector(0, 1),
  "left": Vector(-1, 0),
  "right": Vector(1, 0),
}


class OpenRouterSnakeAgent:
  def __init__(
    self,
    model: str = "meta-llama/llama-3.1-8b-instruct",
    temperature: float = 0.2,
    max_retries: int = 2,
    request_timeout: float = 10,
  ):
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
      raise RuntimeError("OPENROUTER_API_KEY is not set in the environment.")

    self.session = requests.Session()
    self.session.headers.update(
      {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/jack/snake-ai-pytorch",  # advisory header
        "X-Title": "Snake LLM Controller",
      }
    )
    self.model = model
    self.temperature = temperature
    self.max_retries = max(0, max_retries)
    self.request_timeout = max(request_timeout, 1)

  def choose_direction(self, game: SnakeGame) -> str:
    prompt = self._build_prompt(game)
    response_text = self._request_direction(prompt)
    direction = self._parse_direction(response_text)
    print('\n\nprompt\n', prompt, '\noutput', direction)
    if direction:
      return direction
    current_direction = direction_to_text(game.direction)
    if current_direction in DIRECTION_VECTORS:
      return current_direction
    return "up"

  def _request_direction(self, prompt: str) -> str:
    payload = {
      "model": self.model,
      "temperature": self.temperature,
      "messages": [
        {
          "role": "system",
          "content": (
            "You control a snake in the classic snake game. "
            "Always respond with exactly one word: straight, left, or right. "
            "Pick the safest move that helps reach the food."
          ),
        },
        {
          "role": "user",
          "content": prompt,
        },
      ],
    }

    for attempt in range(self.max_retries + 1):
      try:
        response = self.session.post(
          OPENROUTER_URL,
          json=payload,
          timeout=self.request_timeout,
        )
        if response.status_code != 200:
          if attempt >= self.max_retries:
            print(
              f"[OpenRouterSnakeAgent] Request failed with status {response.status_code}: {response.text}"
            )
            return ""
          time.sleep(0.5)
          continue
        body = response.json()
        choice = body.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
          return content
        return ""
      except requests.RequestException as error:
        if attempt >= self.max_retries:
          print(f"[OpenRouterSnakeAgent] Request failed: {error}")
          return ""
        time.sleep(0.5)
    return ""

  def _parse_direction(self, text: str) -> Optional[str]:
    if not text:
      return None

    normalized = text.strip().lower()
    if not normalized:
      return None

    tokens = normalized.replace(".", " ").replace(",", " ").split()
    for token in tokens:
      if token in DIRECTION_VECTORS:
        return token
    return None

  def _build_prompt(self, game: SnakeGame) -> str:
    head = game.head
    food = game.food
    direction_label = direction_to_text(game.direction)
    food_direction = describe_food(head, food)
    collision_report = describe_collisions(game, head)
    snake_length = len(game.snake)
    score = game.score

    prompt_lines = [
      # f"The board is {game.width} wide and {game.height} tall.",
      # f"The snake length is {snake_length} and the score is {score}.",
      f"The head is at ({head.x}, {head.y}) moving {direction_label}.",
      f"The food is {food_direction}.",
      collision_report,
      "Choose the safest move that helps reach the food.",
      "Respond with one word: up, down, left, or right.",
    ]
    return " ".join(prompt_lines)


def direction_to_text(direction: Vector) -> str:
  if direction == Vector(1, 0):
    return "right"
  if direction == Vector(-1, 0):
    return "left"
  if direction == Vector(0, -1):
    return "up"
  if direction == Vector(0, 1):
    return "down"
  return "unknown"


def describe_food(head: Point, food: Point) -> str:
  dx = food.x - head.x
  dy = food.y - head.y

  vertical = ""
  if dy < 0:
    vertical = "top"
  elif dy > 0:
    vertical = "bottom"

  horizontal = ""
  if dx < 0:
    horizontal = "left"
  elif dx > 0:
    horizontal = "right"

  labels = " ".join(part for part in (vertical, horizontal) if part)
  if labels:
    return f"{labels} from the snake's head"
  return "at the snake's head"


def describe_collisions(game: SnakeGame, head: Point) -> str:
  dangers = []
  for label, vector in DIRECTION_VECTORS.items():
    probe_point = Point(head.x + vector.x, head.y + vector.y)
    if game.is_out_of_bounds(probe_point) or game.is_collision(probe_point):
      dangers.append(f"Moving {label} will collide.")
    else:
      dangers.append(f"Moving {label} is safe.")
  return " ".join(dangers)


def apply_move(game: SnakeGame, move: str) -> bool:
  vector = DIRECTION_VECTORS.get(move)
  if vector is None:
    return False

  reverse = Vector(-game.direction.x, -game.direction.y)
  if vector == reverse:
    return False

  game.direction = vector
  return True


def main() -> None:
  agent = OpenRouterSnakeAgent()
  game = SnakeGame(25, 25)
  frontend = SnakeGameMatplotlibFrontend(game, tick_rate=10, play_mode="absolute")

  try:
    while True:
      if frontend.handle_user_input():
        return

      move = agent.choose_direction(game)
      applied = apply_move(game, move)
      if not applied:
        apply_move(game, direction_to_text(game.direction))

      step_reward, game_over, score = game.step()
      frontend.render()

      if game_over:
        print(f"Game over. Score: {score}, last reward: {step_reward}")
        time.sleep(0.75)
        game.reset()
        frontend.resize(game)
  finally:
    frontend.close()


if __name__ == "__main__":
  main()


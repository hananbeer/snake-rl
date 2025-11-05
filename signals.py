from game import SnakeGame, Point, Vector

def get_danger_signals(game: SnakeGame) -> list[bool]:
  return [
    game.is_game_over(game.get_next_position(game.head, game.direction)),
    game.is_game_over(game.get_next_position(game.head, game.get_relative_direction('right'))),
    game.is_game_over(game.get_next_position(game.head, game.get_relative_direction('left'))),
  ]

def get_direction_bitmap_signals(game: SnakeGame) -> list[bool]:
  return [
    game.direction == Vector(-1, 0),
    game.direction == Vector(1, 0),
    game.direction == Vector(0, -1),
    game.direction == Vector(0, 1),
  ]

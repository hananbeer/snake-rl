# Snake

## requirements

- fixed seed


# different designs:

## simple snake game

- fixed grid

## relative controls

### controls:

- world (up, down, left, right)
- local (left / right / no-op)

### signals:

- one-hot world vector (up, down, left, right)
- world direction tuple ([-1..1, -1..1])
- local direction tuple ([-1..1, -1..1])
- food axis direction (-1, 1)
- food distance (-X..X, -Y..Y)
- food world location (x, y)
- snake world location (x, y)


### food

- single food
- multiple instances of food
- multiple kinds of foods, including poisons
- obstacles like walls?


### snake behavior

- non self-colliding
- self-colliding
- wall teleport
- wall is death



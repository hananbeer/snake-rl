from game import SnakeGameHuman
import pygame

if __name__ == '__main__':
    game = SnakeGameHuman()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over:
            print('Score', score)
            game.reset()

    pygame.quit()

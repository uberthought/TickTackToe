import tensorflow as tf
import numpy as np
import os.path
import math

from Game import Game
from network import DNN
from random import randint

game = Game()
dnn = DNN(game.state_size, game.action_size)

wins = 0
ties = 0
illegal = 0
lost = 0
games = 1

for i in range(5000):

    state1 = game.get_state()
    actions1 = dnn.run([state1])

    action = np.argmax(actions1)

    # Take the action (aa) and observe the the outcome state (s′s′) and reward (rr).
    game.move(action, 1)
    score = game.get_score()

    # play other player
    if score != 0 and score != 100:
        moves = game.moves()
        game.move(np.random.choice(moves, 1), -1)
        score = game.get_score()

    if score == 0 or score == 100:
        games = games + 1
        if game.illegal():
            illegal = illegal + 1
            print()
            print(game.state)
            print(actions1)
        elif game.tied():
            ties = ties + 1
        elif game.won():
            wins = wins + 1
        elif game.lost():
            lost = lost + 1

        game = Game()

print('Won: ', wins / games * 100)
print('Lost: ', lost / games * 100)
print('Tied: ', ties / games * 100)
print('Illegal: ', illegal / games * 100)

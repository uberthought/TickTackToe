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
games = 1

for i in range(10000):

    state1 = game.get_state()
    actions1 = dnn.run([state1])

    move_mask = game.move_mask()
    action = np.argmax(actions1 + move_mask * 2)

    # Take the action (aa) and observe the the outcome state (s′s′) and reward (rr).
    game.move(action, 1)
    score = game.get_score()

    # play other player
    if score != -1 and score != 1:
        moves = game.moves()
        if len(moves) > 0:
            game.move(np.random.choice(moves, 1), -1)
            score = game.get_score()

    if score == -1 or score == 1:
        games = games + 1
        if len(game.moves()) == 0:
            ties = ties + 1
        elif score == 1:
            wins = wins + 1
        won = wins  / games * 100
        lost = (games - (wins + ties)) / games * 100
        tied = ties / games * 100
        game = Game()

print('Won: ', won)
print('Lost: ', lost)
print('Tied: ', tied)

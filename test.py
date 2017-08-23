import tensorflow as tf
import numpy as np
import os.path
import math

from Game import Game
from network import DNN
from random import randint

game = Game()
dnn1 = DNN(game.state_size, game.action_size)

wins = 0
ties = 0
illegal = 0
illegal2 = 0
lost = 0
games = 0


def player_turn(dnn, player):
    state = game.get_state()
    actions = dnn.run([state])
    move_mask = Game.move_mask(state)
    actions = actions * move_mask
    action = np.argmax(actions)
    game.move(action, player)
    # print(np.reshape(actions, (3, 3)))


def player_turn_random(player):
    moves = game.moves()
    action = np.random.choice(moves, 1)
    game.move(action, player)


for i in range(10000):

    # print(np.reshape(game.state, (3, 3)))

    # player 1 moves
    player_turn(dnn1, 1)

    # print(np.reshape(game.state, (3, 3)))

    # player 2 moves
    if not game.is_finished():
        player_turn_random(-1)

    if game.is_finished():
        games = games + 1

        if game.illegal():
            illegal = illegal + 1
        elif game.tied():
            ties = ties + 1
        elif game.won():
            wins = wins + 1
        elif game.lost():
            # exit()
            lost = lost + 1

        game = Game()

print('Won: ', wins / games * 100)
print('Lost: ', lost / games * 100)
print('Tied: ', ties / games * 100)
print('Illegal: ', illegal / games * 100)

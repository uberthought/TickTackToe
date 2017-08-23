import tensorflow as tf
import numpy as np
import os.path
import math
import pickle

from Game import Game
from network import DNN
from random import randint

game = Game()
dnn1 = DNN(game.state_size, game.action_size, 'p1')

experiences = []
if os.path.exists('experiences.p'):
    experiences = pickle.load(open("experiences.p", "rb"))


def player_turn(dnn, player):
    state = game.get_state()
    actions = dnn.run([state])
    move_mask = Game.move_mask(state)
    actions = actions * move_mask
    action = np.argmax(actions)

    if randint(0, 10) == 0:
        moves = game.moves()
        action = np.random.choice(moves, 1)

    # Take the action (aa) and observe the the outcome state (s′s′) and reward (rr).
    game.move(action, player)
    return action


def player_turn_random(player):
    moves = game.moves()
    action = np.random.choice(moves, 1)
    game.move(action, player)
    return action


# For life or until learning is stopped...
for i in range(10000):

    state0 = game.get_state()

    # player 1 move
    action1 = player_turn(dnn1, 'p1')

    state1 = game.get_state()
    terminal = game.is_finished()

    if terminal:
        score1 = game.get_score('p1')
        experience1 = {'state0': state0, 'action': action1, 'state1': state1, 'score': score1, 'terminal': terminal}
        experiences.append(experience1)
    else:
        #player 2 move
        action2 = player_turn_random('p2')

        state2 = game.get_state()

        score1 = game.get_score('p1')
        score2 = game.get_score('p2')

        terminal = game.is_finished()

        experience1 = {'state0': state0, 'action': action1, 'state1': state2, 'score': score1, 'terminal': terminal}
        experiences.append(experience1)

    if terminal:
        # switch to next game

        game = Game()

    if i != 0 and i % 100 == 0:
        pickle.dump(experiences, open("experiences.p", "wb"))
        print(i)

import tensorflow as tf
import numpy as np
import os.path
import math
import pickle

from Game import Game
from network import DNN
from random import randint

game = Game()
dnn1 = DNN(game.state_size, game.action_size)

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


def train(dnn, experiences):

    # terminal_experiences = []
    # for experience in experiences:
    #     terminal = experience['terminal']
    #     score = experience['score']
    #     if terminal and score == 0:
    #         terminal_experiences.append(experience)
    #
    # training_experiences = []
    #
    # indices = np.random.choice(len(experiences), 100)
    # for experience in np.take(experiences, indices, axis=0):
    #     training_experiences.append(experience)
    #
    # if len(terminal_experiences) > 0:
    #     indices = np.random.choice(len(terminal_experiences), 100)
    #     for experience in np.take(terminal_experiences, indices, axis=0):
    #         training_experiences.append(experience)

    training_experiences = np.random.choice(experiences, 500)

    X = np.array([], dtype=np.float).reshape(0, game.state_size)
    Y = np.array([], dtype=np.float).reshape(0, game.action_size)

    for experience in training_experiences:
        state0 = experience['state0']
        action = experience['action']
        state1 = experience['state1']
        score = experience['score']
        terminal = experience['terminal']

        actions1 = dnn.run([state0])

        if terminal:
            actions1[0][action] = score
        else:
            actions2 = dnn.run([state1])
            discount_factor = .5
            actions1[0][action] = score + discount_factor * np.max(actions2)

        # move_mask = Game.move_mask(state0)
        # actions1 = actions1 * move_mask

        X = np.concatenate((X, np.reshape(state0, (1, game.state_size))), axis=0)
        Y = np.concatenate((Y, actions1), axis=0)

    dnn.train(X, Y)

print('experiences ', len(experiences))

# For life or until learning is stopped...
for i in range(10000000):

    state0 = game.get_state()

    # player 1 move
    action1 = player_turn(dnn1, 1)

    state1 = game.get_state()
    terminal = game.is_finished()

    if terminal:
        score1 = game.get_score()
        experience1 = {'state0': state0, 'action': action1, 'state1': state1, 'score': score1, 'terminal': terminal}
        experiences.append(experience1)
    else:
        #player 2 move
        action2 = player_turn_random(-1)

        state2 = game.get_state()

        score1 = game.get_score()

        terminal = game.is_finished()

        experience1 = {'state0': state0, 'action': action1, 'state1': state2, 'score': score1, 'terminal': terminal}
        experiences.append(experience1)

    if terminal:
        # switch to next game

        game = Game()

        # train
        train(dnn1, experiences)

    if i != 0 and i % 100 == 0:
        pickle.dump(experiences, open("experiences.p", "wb"))
        print(i)
        dnn1.save()

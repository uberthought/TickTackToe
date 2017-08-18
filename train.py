import tensorflow as tf
import numpy as np
import os.path
import math

from Game import Game
from network import DNN
from random import randint

game = Game()
dnn = DNN(game.state_size, game.action_size)

experiences = []

won = 0
games = 1

# For life or until learning is stopped...
for i in range(10000000):

    state1 = game.get_state()
    actions1 = dnn.run([state1])

    action = np.argmax(actions1)

    # print('state1')
    # print(state1)
    # print('actions1')
    # print(actions1)
    # print('action ', action)

    if randint(0,24) == 0:
        moves = game.moves()
        action = np.random.choice(moves, 1)

    # Take the action (aa) and observe the the outcome state (s′s′) and reward (rr).
    game.move(action, 1)
    score = game.get_score()

    # play other player
    if score != 0 and score != 100:
        moves = game.moves()
        game.move(np.random.choice(moves, 1), -1)

    score = game.get_score()
    state2 = game.get_state()

    # store experience as state1, action, score, state2
    experience = {'state1': state1, 'action': action, 'score': score, 'state2': state2}
    experiences.append(experience)

    if score == 0 or score == 100:
        games = games + 1

        # switch to next game

        game = Game()

        # train

        training_experiences = np.random.choice(experiences, 100)

        X = np.array([], dtype=np.float).reshape(0,game.state_size)
        Y = np.array([], dtype=np.float).reshape(0,game.action_size)

        for experience in training_experiences:
            state1 = experience['state1']
            action = experience['action']
            score = experience['score']
            state2 = experience['state2']

            actions1 = dnn.run([state1])

            # actions1[0][action] = score
            if score == 0 or score == 100:
                actions1[0][action] = score
            else:
                actions2 = dnn.run([state2])
                discount_factor = 0.9
                actions1[0][action] = score + discount_factor * np.max(actions2)

            move_mask = Game.move_mask(state1)
            actions1 = actions1 * move_mask

            # print()
            # print(actions1)

            X = np.concatenate((X, np.reshape(state1, (1, game.state_size))), axis=0)
            Y = np.concatenate((Y, actions1), axis=0)

        dnn.train(X, Y)

    if i % 100 == 0:
        print(len(experiences))
        dnn.save()

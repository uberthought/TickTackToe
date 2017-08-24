import tensorflow as tf
import numpy as np
import os.path
import math
import pickle

from Game import Game
from network import DNN
from random import randint

game = Game()
dnn = DNN(game.state_size, game.action_size)

old_experiences = []
if os.path.exists('old_experiences.p'):
    old_experiences = pickle.load(open("old_experiences.p", "rb"))


def train(dnn, experiences):
    X = np.array([], dtype=np.float).reshape(0, game.state_size)
    Y = np.array([], dtype=np.float).reshape(0, game.action_size)

    for experience in experiences:
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
            discount_factor = 1
            actions1[0][action] = score + discount_factor * np.max(actions2)

        X = np.concatenate((X, np.reshape(state0, (1, game.state_size))), axis=0)
        Y = np.concatenate((Y, actions1), axis=0)

    return dnn.train(X, Y)


print('old_experiences ', len(old_experiences))

games = 0

# For life or until learning is stopped...
for i in range(10000000):

    loss = train(dnn, old_experiences)
    dnn.save()

    print('i ', i)
    print('loss ', loss)


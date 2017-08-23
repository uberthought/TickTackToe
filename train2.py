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


def train(dnn, experiences):

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

        X = np.concatenate((X, np.reshape(state0, (1, game.state_size))), axis=0)
        Y = np.concatenate((Y, actions1), axis=0)

    dnn.train(X, Y)

print('experiences ', len(experiences))

# For life or until learning is stopped...
for i in range(10000000):
    # train
    train(dnn1, experiences)

    if i != 0 and i % 100 == 0:
        print(i)
        dnn1.save()

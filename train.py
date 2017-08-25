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
old_experiences = []
if os.path.exists('old_experiences.p'):
    old_experiences = pickle.load(open("old_experiences.p", "rb"))


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


def add_experience(experience):

    index = len(old_experiences) - 1
    for existing in reversed(old_experiences):
        if (existing['state0'] == experience['state0']).all():
            old_experiences.pop(index)
        index -= 1
    old_experiences.append(experience)

    experiences.append(experience)


print('old_experiences ', len(old_experiences))

games = 0

# For life or until learning is stopped...
for i in range(10000000):

    state0 = game.get_state()

    # player 1 move
    action1 = player_turn(dnn1, 1)

    state1 = game.get_state()
    terminal = game.is_finished()

    if terminal:
        score1 = game.get_score()
        experience = {'state0': state0, 'action': action1, 'state1': state1, 'score': score1, 'terminal': terminal}
        add_experience(experience)
    else:
        #player 2 move
        action2 = player_turn_random(-1)

        state2 = game.get_state()

        score1 = game.get_score()

        terminal = game.is_finished()

        experience = {'state0': state0, 'action': action1, 'state1': state2, 'score': score1, 'terminal': terminal}
        add_experience(experience)

    if terminal:

        # switch to next game

        games += 1
        game = Game()

        if randint(0, 2) == 0:
            game.move(randint(0, 8), -1)

        # add old experiences

        if len(old_experiences) >= 0:
            random_old_experiences = np.random.choice(old_experiences, len(experiences) * 4).tolist()
            experiences = experiences + random_old_experiences

        # train
        new_loss = train(dnn1, experiences)

        print('games ', games)

        experiences = []

        pickle.dump(old_experiences, open("old_experiences.p", "wb"))
        dnn1.save()

import tensorflow as tf
import numpy as np
import os.path
import math

from Game import Game
from network import DNN
from random import randint

def player_turn(self, player):
    state = self.game.get_state()
    actions = self.dnn.run([state])
    move_mask = Game.move_mask(state)
    actions = actions * move_mask
    action = np.argmax(actions)
    self.game.move(action, player)


def player_turn_random(self, player):
    moves = self.game.moves()
    action = np.random.choice(moves, 1)
    self.game.move(action, player)

class Test:
    def __init__(self):
        self.message = 'hello world'
        self.game = Game()
        self.dnn = DNN()

    def run(self):
        games = 0
        illegal = 0
        ties = 0
        wins = 0
        lost = 0

        for i in range(2000):
            player_turn(self, 1)

            if not self.game.is_finished():
                player_turn_random(self, -1)

            if self.game.is_finished():
                games = games + 1

                if self.game.illegal():
                    illegal = illegal + 1
                elif self.game.tied():
                    ties = ties + 1
                elif self.game.won():
                     wins = wins + 1
                elif self.game.lost():
                    lost = lost + 1

                self.game = Game()
                if randint(0, 2) == 0:
                    self.game.move(randint(0, 8), -1)

        result = 'Won ' + str(wins / games * 100) + '<br>'
        result += 'Lost ' + str(lost / games * 100) + '<br>'
        result += 'Tied ' + str(ties / games * 100) + '<br>'
        result += 'Illegal ' + str(illegal / games * 100) + '<br>'
        return result

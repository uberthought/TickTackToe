import numpy as np
import os.path
import math


class Game:
    def __init__(self):
        self.illegal_move = None
        self.state_size = 9
        self.action_size = 9
        self.state = np.zeros(9)

    def get_state(self):
        return np.copy(self.state)

    def get_score(self):
        if self.illegal_move:
            return -1
        moves = np.where(self.state == 0)[0]
        if len(moves) == 0:
            return 1
        foo = np.reshape(self.state, (3, 3))
        rows = np.sum(foo, axis=0)
        columns = np.sum(foo, axis=1)
        diagonal1 = foo[0][0] + foo[1][1] + foo[2][2]
        diagonal2 = foo[0][2] + foo[1][1] + foo[2][0]
        all = np.concatenate((rows, columns, [diagonal1], [diagonal2]))
        if all.max() == 3:
            return 1
        elif all.min() == -3:
            return -1
        return 0.9

    def move(self, x, v):
        if self.state[x] != 0:
            self.illegal_move = True
        else:
            self.state[x] = v

    def moves(self):
        return np.where(self.state == 0)[0]

    def move_mask(self):
        return np.where(self.state == 0, 0, -1)

    def invert(self):
        self.state = self.state * -1
import numpy as np
import os.path
import math


class Game:
    def __init__(self):
        self.illegal_move = None
        self.state_size = 18
        self.action_size = 9
        self.state = np.zeros(9)

    def get_state(self):
        # return np.copy(self.state)
        p1 = np.where(self.state == 1, 1, 0)
        p2 = np.where(self.state == -1, 1, 0)
        return np.concatenate((p1, p2))

    def get_score(self):

        if self.illegal():
            return 0

        if self.tied():
            return 100

        foo = np.reshape(self.state, (3, 3))
        rows = np.sum(foo, axis=0)
        columns = np.sum(foo, axis=1)
        diagonal1 = foo[0][0] + foo[1][1] + foo[2][2]
        diagonal2 = foo[0][2] + foo[1][1] + foo[2][0]
        all = np.concatenate((rows, columns, [diagonal1], [diagonal2]))

        if all.max() == 3:
            return 100
        elif all.min() == -3:
            return 0

        return 10

    def move(self, x, v):
        if self.state[x] != 0:
            self.illegal_move = True
        else:
            self.state[x] = v

    def moves(self):
        return np.where(self.state == 0)[0]

    def won(self):
        return not self.illegal() and not self.tied() and self.get_score() == 100

    def lost(self):
        return not self.illegal() and not self.tied() and self.get_score() == 0

    def illegal(self):
        return self.illegal_move

    def tied(self):
        return len(self.moves()) == 0

    def move_mask(x):
        p1 = x[:9]
        p2 = x[9:]
        foo = p1 + p2
        return np.where(foo != 0, 0, 1)
import numpy as np
import os.path
import math


class Game:
    def __init__(self):
        self.state = np.zeros((3,3))
        self.illegal_move = None
        self.turn = -1

    def get_state(self):
        return np.copy(np.reshape(self.state, (1, -1)))

    def get_score(self):
        if self.illegal_move:
            return -1
        rows = np.sum(self.state, axis=0)
        columns = np.sum(self.state, axis=1)
        diagonal1 = self.state[0][0] + self.state[1][1] + self.state[2][2]
        diagonal2 = self.state[0][2] + self.state[1][1] + self.state[2][0]
        all = np.concatenate((rows, columns, [diagonal1], [diagonal2]))
        if all.max() == 3:
            return 1
        elif all.min() == -3:
            return -1
        return 0.5

    def move(self, x, y, v):
        if self.state[x][y] != 0:
            self.illegal_move = True
        else:
            self.state[x][y] = v

        self.turn = -self.turn

        return self.get_state(), self.get_score()
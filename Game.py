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
        p1 = np.where(self.state == 1, 1, 0)
        p2 = np.where(self.state == -1, 1, 0)
        return np.concatenate((p1, p2))

    def get_score(self):

        # player made illegal move
        if self.illegal():
            return 0

        # tied
        if self.tied():
            return 100

        # score
        foo = np.reshape(self.state, (3, 3))
        rows = np.sum(foo, axis=0)
        columns = np.sum(foo, axis=1)
        diagonal1 = foo[0][0] + foo[1][1] + foo[2][2]
        diagonal2 = foo[0][2] + foo[1][1] + foo[2][0]
        all = np.concatenate((rows, columns, [diagonal1], [diagonal2]))

        if all.max() == 3:
            return 0
        elif all.min() == -3:
            return 0

        return 1

    def move(self, x, player):
        if self.state[x] != 0:
            self.illegal_move = player
        else:
                self.state[x] = player

    def moves(self):
        return np.where(self.state == 0)[0]

    def won(self):
        if self.illegal() or self.tied():
            return False

        foo = np.reshape(self.state, (3, 3))
        rows = np.sum(foo, axis=0)
        columns = np.sum(foo, axis=1)
        diagonal1 = foo[0][0] + foo[1][1] + foo[2][2]
        diagonal2 = foo[0][2] + foo[1][1] + foo[2][0]
        all = np.concatenate((rows, columns, [diagonal1], [diagonal2]))

        if all.max() == 3:
            return True

        return False

    def lost(self):
        return not self.illegal() and not self.tied() and not self.won()

    def illegal(self):
        return self.illegal_move

    def tied(self):
        return np.all(self.state != 0)

    def move_mask(x):
        p1 = x[:9]
        p2 = x[9:]
        foo = p1 + p2
        return np.where(foo != 0, 0, 1)

    def is_finished(self):

        # illegal move
        if self.illegal():
            return True

        # tied
        if self.tied():
            return True

        score = self.get_score()
        if  score == 100 or score == 0:
            return True

        return False

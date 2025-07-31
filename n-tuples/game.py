import random
import numpy as np

UP, RIGHT, DOWN, LEFT = range(4)
action_name = ["UP", "RIGHT", "DOWN", "LEFT"]

class IllegalAction(Exception):
    pass

class GameOver(Exception):
    pass

class Board:
    def __init__(self, board=None):
        self.board = np.zeros((16), dtype=int) if board is None else np.array(board, dtype=int)
        self.spawn_tile()   

    def spawn_tile(self):
        empty = [i for i in range(16) if self.board[i] == 0]
        if not empty:
            return
        i = random.choice(empty)
        self.board[i] = 2 if random.random() < 0.9 else 4

    def copyboard(self):
        return np.copy(self.board).reshape((16))

    def rotate_board(self, times):
        return np.rot90(self.board.reshape(4, 4), -times)

    def act(self, action):
        rotated = np.rot90(self.board.reshape(4, 4), -action)
        moved, score = self._move_left(rotated)
        if moved is None:
            raise IllegalAction()
        
        self.board = np.rot90(moved, action).flatten()
        self.spawn_tile()
        return score

    def _move_left(self, board):
        score = 0
        new_board = np.zeros_like(board)
        moved = False

        for row_no, row in enumerate(board):
            new_row = [n for n in row if n != 0]

            i = 0
            while i < len(new_row) - 1:
                if new_row[i] == new_row[i + 1]:
                    new_row[i] *= 2
                    score += new_row[i]
                    new_row[i + 1] = 0
                    i += 2
                else:
                    i += 1

            new_row = [n for n in new_row if n != 0]
            new_row += [0] * (4 - len(new_row))
            new_row = np.array(new_row)

            if not np.array_equal(new_row, row):
                moved = True

            new_board[row_no] = new_row

        return (new_board, score) if moved else (None, None)

                    
    def is_game_over(self):
        if 0 in self.board:
            return False
        board=self.copyboard().reshape(4, 4)
        for action in range(4):
            rotated = np.rot90(self.board, -action)
            moved, score = self._move_left(rotated)
            if moved is not None:
                return False
        return True
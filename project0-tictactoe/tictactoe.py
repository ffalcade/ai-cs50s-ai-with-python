"""
Tic Tac Toe Player
"""

import math, copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY]
    ]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    count_x = 0
    count_o = 0

    for line in board:
        for cell in line:
            if cell == X:
                count_x += 1
            elif cell == O:
                count_o += 1

    return X if count_x <= count_o else O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    available_moves = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                available_moves.add((i, j))
    
    return available_moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action

    if not (0 <= i < 3 and 0 <= j < 3):
        raise IndexError("Action indices must be in range 0 to 2")

    if board[i][j] != EMPTY:
        raise ValueError("Move not allowed: cell is not empty")
    
    new_board = copy.deepcopy(board)
    new_board[i][j] = player(board)

    return new_board

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # check lines
    for line in board:
        if line[0] == line[1] == line[2] != EMPTY:
            return line[0]
    
    # check columns
    for column in range(3):
        if board[0][column] == board[1][column] == board[2][column] != EMPTY:
            return board[0][column]

    # check X1
    if (board[0][0] == board[1][1] == board[2][2] != EMPTY):
        return board[0][0]

    # check X2
    if (board[0][2] == board[1][1] == board[2][0] != EMPTY):
        return board[0][2]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True

    # check empty
    for line in board:
        if EMPTY in line:
            return False

    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    current = player(board)
    maximizing = current == X

    def value(state, alpha, beta, maximizing):
        if terminal(state):
            return utility(state), None
        v = -math.inf if maximizing else math.inf
        best_action = None

        for action in actions(state):
            new_val, _ = value(result(state, action), alpha, beta, not maximizing)
            if maximizing:
                if new_val > v:
                    v = new_val
                    best_action = action
                alpha = max(alpha, v)
            else:
                if new_val < v:
                    v = new_val
                    best_action = action
                beta = min(beta, v)
            if beta <= alpha:
                break

        return v, best_action

    _, action = value(board, -math.inf, math.inf, maximizing)
    return action

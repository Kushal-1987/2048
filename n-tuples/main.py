import numpy as np
import pickle
from pathlib import Path
from game import Board, IllegalAction, GameOver
from agent import nTupleNewrok
from collections import namedtuple

Transition = namedtuple("Transition", "s, a, r, s_after, s_next")
Gameplay = namedtuple("Gameplay", "transition_history game_reward max_tile")

def play(agent, board, spawn_random_tile=False, display=False, delay=0.3):
    import time
    b = Board(board)
    r_game = 0
    transition_history = []

    while True:
        if display:
            # Optional: Implement display logic here or in GUI
            pass

        # TODO: Modify this call to implement epsilon-greedy exploration
        a_best = agent.best_action(b.board)

        s = b.copyboard()
        try:
            r = b.act(a_best)
            r_game += r
            s_after = b.copyboard()
            b.spawn_tile()
            s_next = b.copyboard()
            transition_history.append(
                Transition(s=s, a=a_best, r=r, s_after=s_after, s_next=s_next)
            )
        except IllegalAction:
            if b.is_game_over():
                if display:
                    print("Game Over!")
                break
            # TODO: Handle illegal action without ending the game prematurely
            pass
        except GameOver:
            if display:
                print("Game Over!")
            break

    gp = Gameplay(
        transition_history=transition_history,
        game_reward=r_game,
        max_tile=2 ** max(b.board.flatten()),
    )
    
    return gp

def train(agent, episodes=1000, display=False):
    for episode in range(episodes):
        board = np.zeros((16), dtype=int)
        # TODO: Spawn initial tiles on the board before starting the game
        gp = play(agent, board, spawn_random_tile=True, display=display)

        # TODO: Add logging or evaluation every N episodes
        if episode % 100 == 0:
            print(f"Episode {episode} finished. Reward: {gp.game_reward}")

if __name__ == "__main__":
    TUPLES = [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(3, 0), (3, 1), (3, 2), (3, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(0, 3), (1, 3), (2, 3), (3, 3)],
    ]

    agent = nTupleNewrok(TUPLES)

    # TODO: Implement epsilon decay schedule or fixed epsilon in agent.best_action()

    train(agent, episodes=1000, display=True)
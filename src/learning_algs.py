import numpy as np
import chess
from chess.pgn import Game
from agent import Agent




class Q_learn(object):
    def __init__(self, agent, env, memorySize=1000):
        self.agent = agent
        self.env = env
        self.memsize = memorySize
        self.memory = []
        self.reward_trace = []
        self.samp_probabilities = []

    def learn(self, iterations=100, updateThreshold=10):
        for x in range(iterations):
            if x % updateThreshold == 0:
                print("iteration: ", x)
                self.agent.freeze_model()
            greedy = True if x == iterations - 1 else False
            self.env.reset()
            self.play(x, greedy=greedy)

        pgn = Game.from_board(self.env.board)
        return pgn
    
    def play(self, explorationRate, greedy=False, maxMoves=None):
        episodeEnd = False
        turnNumber = 0
        epsilonGreedy = max(0.05, 1 / (1 + (explorationRate / 250))) if not(greedy) else 0.0
        while not(episodeEnd):
            state = Agent.one_hot_encode(self.env.board, chess.WHITE)
            explore = np.random.uniform(0,1) < epsilonGreedy
            if explore:
                move = self.env.random_action()
                original_pos = move.from_square
                dest_pos = move.to_square
            else:
                action_values = self.agent.find_move(np.expand_dims(state, axis=0))
                action_values = np.reshape(np.squeeze(action_values), (64,64))
                action_space = self.env.moves_to_action_sapace()
                action_values = np.multiply(action_space, action_values)
                move_from = np.argmax(action_values, axis=None) // 64
                move_to = np.argmax(action_values, axis=None) % 64
                moves = [x for x in self.env.board.legal_moves if x.from_square == move_from and x.to_square == move_to]
                if len(moves) == 0:  # If all legal moves have negative action value, explore.
                    move = self.env.get_random_action()
                    move_from = move.from_square
                    move_to = move.to_square
                else:
                    move = np.random.choice(moves)  # If there are multiple max-moves, pick a random one.








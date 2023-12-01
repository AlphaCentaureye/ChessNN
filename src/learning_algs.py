import numpy as np
from chess.pgn import Game




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
            
            break








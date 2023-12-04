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

    def learn(self, iterations=100, updateThreshold=10, explorationRateRatio=250):
        for x in range(iterations):
            if x % updateThreshold == 0:
                print("iteration: ", x)
                self.agent.freeze_model()
            greedy = True if x == iterations - 1 else False
            self.env.reset()
            self.play(x, greedy=greedy, explorationRateRatio=explorationRateRatio)

        pgn = Game.from_board(self.env.board)
        return pgn
    
    def play(self, explorationRate, greedy=False, maxMoves=300, explorationRateRatio=250):
        # max moves is defaulted to 300 as that should never interfere normally, but should prevent it from going on too long in initial training
        keep_going = True
        turnNumber = 0
        epsilonGreedy = max(0.05, 1 / (1 + (explorationRate / explorationRateRatio))) if not(greedy) else 0.0
        while keep_going:
            state = Agent.one_hot_encode(self.env.board, chess.WHITE) # white for now
            explore = np.random.uniform(0,1) < epsilonGreedy
            if explore:
                move = self.env.random_action()
                move_from = move.from_square
                move_to = move.to_square
            else:
                action_values = self.agent.find_move(state)
                action_values = np.reshape(np.squeeze(action_values), (64,64))
                action_space = self.env.moves_to_action_space()
                action_values = np.multiply(action_space, action_values)
                # argmax with axis none gives the index of the maximum value as if the array was flattened so this gives the board tile positions
                move_from = np.argmax(action_values, axis=None) // 64
                move_to = np.argmax(action_values, axis=None) % 64
                movePromote = chess.Move.from_uci(chess.square_name(move_from)+chess.square_name(move_to)+'q')
                moveNormal = chess.Move.from_uci(chess.square_name(move_from)+chess.square_name(move_to))
                if movePromote in self.env.board.legal_moves:
                    move = movePromote
                elif moveNormal in self.env.board.legal_moves:
                    move = moveNormal
                else:
                    # this should never be called but it's jsut in case to prevent an error
                    move = self.env.random_action()
                    move_from = move.from_square
                    move_to = move.to_square

            keep_going, reward = self.env.step(move, doRandomMove=False, staticAgent=self.agent)
            new_state = Agent.one_hot_encode(self.env.board, chess.WHITE) # white for now
            if len(self.memory) > self.memsize:
                self.memory.pop(0)
                self.samp_probabilities.pop(0)
            turnNumber += 1
            if turnNumber > maxMoves:
                keep_going = False
                reward = 0
            if not(keep_going):
                new_state = new_state * 0 # reset everything to 0
            self.memory.append([state, (move_from, move_to), reward, new_state])
            self.samp_probabilities.append(1)
            self.reward_trace.append(reward)
            self.update_agent(turnNumber)
        try:
            display(self.env.board)
        except:
            print(self.env.board)
        return self.env.board
    

    # grab random moves from memory
    def sample_memory(self, turncount):
        minibatch = []
        memory = self.memory[:-turncount] # first 'turncount' many items from memory
        probs = self.samp_probabilities[:-turncount]
        sample_probs = [probs[n] / np.sum(probs) for n in range(len(probs))]
        # retruns random indices, amount in memory or 1028 of them whichever is smaller, same one can be picked twice, and probability of getting them is weighted
        indices = np.random.choice(range(len(memory)), min(1028, len(memory)), replace=True, p=sample_probs)
        for i in indices:
            minibatch.append(memory[i])
        return minibatch, indices


    def update_agent(self, turncount):
        if turncount < len(self.memory):
            minibatch, indices = self.sample_memory(turncount)
            temp_diff_errors = self.agent.update_network(minibatch)
            for n, i in enumerate(indices):
                self.samp_probabilities[i] = np.abs(temp_diff_errors[n])








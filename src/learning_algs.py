import numpy as np
import chess
from chess.pgn import Game
from agent import Agent
import os
try:
    from google.colab import files
except Exception as e:
    print("couldn't import colab files library:", e)




class Q_learn(object):
    def __init__(self, agent, env, memorySize=1000):
        self.agent = agent
        self.env = env
        self.memsize = memorySize
        self.memory = []
        self.gameMemory = []
        self.reward_trace = []
        self.samp_probabilities = []

    def learn(self, iterations=100, updateThreshold=10, maxMoves=150, explorationRateRatio=250, explRtOffset = 0, backupRate=10, display=False, randMove=False, waitForWins = 2):
        wins = 0
        randomMoves = True
        self.agent.freeze_model()
        for x in range(iterations):
            print("iteration: ", x)
            if backupRate != 0 and x % abs(backupRate) == 0 and x != 0:
                try:
                    self.agent.saveNN(os.path.join('/content/savedNNs/', str(x)))
                    if backupRate > 0:
                        try:
                            files.download(os.path.join('/content/savedNNs/', str(x), 'chessNN_model.zip'))
                            print('model instance downloaded')
                        except Exception as e:
                            print('1:', e)
                except Exception as e:
                    print('2:', e)
            greedy = True if x == iterations - 1 else False
            self.env.reset()
            if randomMoves and updateThreshold < 0:
                boardState = self.play(x, greedy=greedy, maxMoves=maxMoves, explorationRateRatio=explorationRateRatio, explRtOffset=explRtOffset, displayBoard=display, randMove=True)
            else:
                boardState = self.play(x, greedy=greedy, maxMoves=maxMoves, explorationRateRatio=explorationRateRatio, explRtOffset=explRtOffset, displayBoard=display, randMove=randMove)

            if boardState.result() == '1-0':
                wins += 1
            if updateThreshold < 0 and waitForWins == wins:
                self.agent.freeze_model()
                wins = 0
                randomMoves = False
            elif x % updateThreshold == 0 and updateThreshold >= 0:
                self.agent.freeze_model()

        pgn = Game.from_board(self.env.board)
        return pgn
    
    def play(self, explorationRate, greedy=False, maxMoves=150, explorationRateRatio=250, explRtOffset=0, displayBoard=False, randMove=False):
        # max moves is defaulted to 300 as that should never interfere normally, but should prevent it from going on too long in initial training
        keep_going = True
        turnNumber = 0
        epsilonGreedy = max(0.05, 1 / (1 + ((explorationRate+explRtOffset) / explorationRateRatio))) if not(greedy) else 0.0
        while keep_going:
            print(explorationRate)
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
                try:
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
                except:
                    move = self.env.random_action()
                    move_from = move.from_square
                    move_to = move.to_square

            keep_going, reward = self.env.step(move, doRandomMove=randMove, staticModel=self.agent.frozen_model, displayBoard=displayBoard)
            new_state = Agent.one_hot_encode(self.env.board, chess.WHITE) # white for now
            if len(self.memory) > self.memsize:
                self.memory.pop(0)
                self.samp_probabilities.pop(0)
            if len(self.gameMemory) > (self.memsize//10)*2:
                self.gameMemory = self.gameMemory[10:]
            turnNumber += 1
            if turnNumber > maxMoves:
                keep_going = False
                reward = 0
            self.memory.append([state, (move_from, move_to), reward, new_state])
            self.gameMemory.append([state, (move_from, move_to), reward])
            self.samp_probabilities.append(1)
            #self.reward_trace.append(reward)
            self.update_agent(turnNumber)

        return self.env.board
    

    # grab random moves from memory
    def sample_memory(self, turncount):
        minibatch = []
        memory = self.memory[:-turncount] # first 'turncount' many items from memory
        probs = self.samp_probabilities[:-turncount]
        probs_sum = np.sum(probs)
        sample_probs = [probs[n] / probs_sum for n in range(len(probs))]
        # retruns random indices, amount in memory or 1028 of them whichever is smaller, same one can be picked twice, and probability of getting them is weighted
        print(len(sample_probs), " : ", np.sum(probs), ' : ', turncount)
        indices = np.random.choice(range(len(memory)), min(128, len(memory)), replace=True, p=sample_probs)
        for i in indices:
            minibatch.append(memory[i])
        return minibatch, indices


    def update_agent(self, turncount):
        if turncount < len(self.memory):
            minibatch, indices = self.sample_memory(turncount)
            temp_diff_errors = self.agent.update_network(minibatch, self.gameMemory)
            for n, i in enumerate(indices):
                self.samp_probabilities[i] = np.abs(temp_diff_errors[n])








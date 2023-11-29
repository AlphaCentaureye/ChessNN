import tensorflow as tf
import os
import chess
from environment import Board
from agent import Agent

agent = Agent("w")
board = Board()
agent.init_network()
#agent.model.summary()
agent.one_hot_encode(board.board)

#agent.saveNN()
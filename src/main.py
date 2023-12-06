import os
import tensorflow as tf
import chess
from environment import Board
from agent import Agent
from learning_algs import Q_learn
import numpy as np



# train network

environment = Board()
agent = Agent(verbose=1)
agent.init_network()
#print(agent.model.summary())

learn = Q_learn(agent, environment)
learn.learn(iterations=30, updateThreshold=1, explorationRateRatio=5, display=True)

try:
    agent.saveNN() # optional path param, but defaults to path /content/savedNNs/
except Exception as e:
    print(e)



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


for x in range(50):
    learn.learn(iterations=50, updateThreshold=2, explRtOffset=x*10, explorationRateRatio=300, display=False)

    try:
        agent.model.save("/" + str(x) + "/")
    except Exception as e:
        print(e)



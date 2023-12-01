import numpy as np




class Q_learn(object):
    def __init__(self, agent, env, memorySize=1000):
        self.agent = agent
        self.env = env
        self.memsize = memorySize
        self.memory = []
        self.reward_trace = []
        self.samp_probabilities = []

    def learn(self):
        pass
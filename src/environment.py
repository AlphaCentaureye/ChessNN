import chess
import numpy as np


class Board(object):
  def __init__(self, FEN=None):
    self.FEN = FEN
    self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
    self.init_action_space()

  def init_action_space(self):
    self.action_space = np.zeros((64, 64))

  def step(self, action):
    
    return

  def random_action(self):
    legal_moves = [x for x in self.board.legal_moves]
    return np.random.choice(legal_moves)

  def moves_to_action_space(self):
    self.init_action_space()
    moves = [[x.from_square, x.to_square] for x in self.board.legal_moves]
    for m in moves:
      self.action_space[m[0], m[1]] = 1
    return self.action_space

  def play(self, action):
    if not(action in self.board.legal_moves):
      return False
    else:
      try:
        self.board.push(action)
        try:
          display(self.board)
        except:
          print(self.board)
        return True
      except Exception as e:
        print(e)
        return False
      
  def reset(self, FEN=None):
    self.FEN = FEN
    self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
    self.init_action_space()




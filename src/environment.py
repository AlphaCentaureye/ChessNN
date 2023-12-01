import chess
import numpy as np
from agent import Agent


class Board(object):
  def __init__(self, FEN=None):
    self.FEN = FEN
    self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
    self.init_action_space()

  def init_action_space(self):
    self.action_space = np.zeros((64, 64))

  def step(self, action, doRandomMove=True):
    board_value_before = self.get_board_value()
    self.board.push(action)
    board_value_after = self.get_board_value()
    reward = board_value_before - board_value_after
    if self.board.result() == '*':
      if doRandomMove:
        random_move = self.random_action()
        self.board.push(random_move)
        if self.board.result() == '*':
          keep_going = True
        else:
          keep_going = False
      else:
        pass
    else:
      keep_going = False
    if self.board.is_game_over():
      reward = 0
      keep_going = False
    return keep_going, reward
          


  def random_action(self):
    legal_moves = [x for x in self.board.legal_moves]
    return np.random.choice(legal_moves)

  def moves_to_action_space(self):
    self.init_action_space()
    moves = [[x.from_square, x.to_square] for x in self.board.legal_moves]
    for m in moves:
      self.action_space[m[0], m[1]] = 1
    return self.action_space
  
  def get_board_value(self):
    # sum up values of pieces on board
    vector = Agent.one_hot_encode(self.board, chess.WHITE)

    pawns = 1 * np.sum(vector[0, :, :])
    rooks = 5 * np.sum(vector[1, :, :])
    knight_and_bishop = 3 * np.sum(vector[2:4, :, :])
    queen = 9 * np.sum(vector[4, :, :])
    return pawns + rooks + knight_and_bishop + queen


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




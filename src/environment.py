import chess
import numpy as np
from agent import Agent

PIECE_INDEX_DICT = {"p":0,
                    "r":1,
                    "n":2,
                    "b":3,
                    "q":4,
                    "k":5,}

class Board(object):
  def __init__(self, FEN=None, reward_factor=0.15):
    self.FEN = FEN
    self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
    self.init_action_space()
    self.rew_mult = reward_factor
    self.previousPos = np.zeros(shape=(6,8,8))
    self.updateState = True

  def init_action_space(self):
    self.action_space = np.zeros((64, 64))

  def step(self, action, doRandomMove=True, staticModel=None, networkColor=chess.WHITE, displayBoard=False):
    board_value_before = self.get_board_value()
    self.board.push(action)
    repeatPunish = int((self.previousPos == self.encodeState(self.board)).all()) * -2
    isCheck = self.board.is_check()
    if self.updateState:
      self.saveState()
    self.updateState = not(self.updateState)
    # print board after bot move
    if displayBoard:
      try:
        display(self.board)
      except:
        print(self.board)
    if self.board.result() == '*':
      if doRandomMove:
        self.board.push(self.random_action())
      else:
        move = Agent.one_hot_decode(staticModel.predict(np.expand_dims(Agent.one_hot_encode(self.board, not(networkColor)), axis=0)), self.board)
        if move:
          self.board.push(move)
        else:
          self.board.push(self.random_action())
      if self.board.result() == '*':
        keep_going = True
      else:
        keep_going = False
      # print board after opponent move
      if displayBoard:
        try:
          display(self.board)
        except:
          print(self.board)
    else:
      keep_going = False
    board_value_after = self.get_board_value()
    reward = (board_value_after - board_value_before + repeatPunish + isCheck - 2*self.board.is_check()) * self.rew_mult
    if self.board.is_game_over():
      result = self.board.result()
      if result == '1-0':
        reward += (2 * networkColor - 1) * 27 * self.rew_mult
      elif result == '0-1':
        reward -= (2 * networkColor - 1) * 27 * self.rew_mult
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
  
  def saveState(self, color=chess.WHITE):
    self.previousPos = np.zeros(shape=(6,8,8))
    for square in range(64):
      piece = str(self.board.piece_at(square))
      if piece != "None":
        self.previousPos[PIECE_INDEX_DICT[piece.lower()]][7-square//8][square%8] = (int(piece.isupper()) if color else int(piece.islower()))
        
  def encodeState(self, color=chess.WHITE):
    vector = np.zeros(shape=(6,8,8))
    for square in range(64):
      piece = str(self.board.piece_at(square))
      if piece != "None":
        vector[PIECE_INDEX_DICT[piece.lower()]][7-square//8][square%8] = (int(piece.isupper()) if color else int(piece.islower()))
    return vector


  def play(self, action):
    if action in self.board.legal_moves:
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
    return False
      
  def reset(self, FEN=None):
    self.FEN = FEN
    self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
    self.init_action_space()




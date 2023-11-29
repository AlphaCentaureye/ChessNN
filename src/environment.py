import chess


class Board(object):
  def __init__(self):
    self.board = chess.Board()


  def play(self, move_string):
    move = chess.Move.from_uci(move_string)
    if not(move in self.board.legal_moves):
      return False
    else:
      try:
        self.board.push(move)
        try:
          display(self.board)
        except:
          print(self.board)
        return True
      except Exception as e:
        print(e)
        return False
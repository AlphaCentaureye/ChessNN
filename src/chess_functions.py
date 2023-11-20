import chess

def play(move_string, board):
  move = chess.Move.from_uci(move_string)
  if not(move in board.legal_moves):
    return False
  else:
    try:
      board.push(move)
      try:
        display(board)
      except:
        print(board)
      return True
    except Exception as e:
      print(e)
      return False
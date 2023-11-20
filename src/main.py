import tensorflow as tf
import chess
import chess_functions as cfn
import nn_functions as nnfn

board = chess.Board()
print(board)


model = nnfn.define_model()
model.summary()
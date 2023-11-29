import tensorflow as tf
import zipfile
import os
import numpy as np
import chess
from math import floor, ceil


PIECE_INDEX_DICT = {"p":0,
                    "r":1,
                    "n":2,
                    "b":3,
                    "q":4,
                    "k":5,}

TILE_INDEX = [*"abcdefgh"] # split characters into array of the characters

class Agent(object):
  def __init__(self, color="w"):
    if color == "w":
      self.color = chess.WHITE
    else:
      self.color = chess.BLACK
    self.isWhite = self.color


  def init_network(self):
      # define model
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.InputLayer(input_shape=(8, 8, 8), name="input_layer"))
      model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=1, activation='relu', name="block1_conv1"))
      model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu', name="block1_conv2"))
      model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu', name="block1_conv3"))
      model.add(tf.keras.layers.Resizing(height=128, width=128, interpolation='bilinear', crop_to_aspect_ratio=False))
      model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation='relu', name="block2_conv1"))
      model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu', name="block2_conv2"))
      model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu', name="block2_conv3"))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu', name="block3_conv1"))
      model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=1, activation='relu', name="block3_conv2"))
      model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=1, activation='relu', name="block3_conv3"))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5)))
      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
      model.add(tf.keras.layers.Dense(units=1024, activation='relu',))
      model.add(tf.keras.layers.Dense(36, activation='softmax', name="output_layer"))

      # compile model
      opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
      model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

      self.model = model

  def test_gpu(self):
      print(tf.test.gpu_device_name())

  def saveNN(self):
    path = os.path.join(os.getcwd(), '/savedNNs/nn_model')
    self.model.save(path)
    with zipfile.ZipFile("/savedNNs/chessNN_model.zip", 'w') as zip_ref:
      zip_ref.write("/savedNNs/chessNN_model")

  def loadNN(self):
    path = '/savedNNs/nn_model'
    with zipfile.ZipFile("/savedNNs/chessNN_model.zip", 'r') as zip_ref:
      zip_ref.extractall()
    self.model = tf.keras.models.load_model(path)


  def trainModel(self, numGames):
      pass

  def one_hot_encode(self, boardState):
    vector = np.zeros(shape=(8,8,8))
    for square in range(64):
      piece = str(boardState.piece_at(square))
      if piece != "None":
        vector[PIECE_INDEX_DICT[piece.lower()]][7-square//8][square%8] = int(piece.isupper()) * 2 - 1
    
    if boardState.turn == self.color:
      vector[6, :, :] = 1 / boardState.fullmove_number
    if boardState.can_claim_draw():
      vector[7, :, :] = 1
    
    return vector
  
  def one_hot_decode(self, vector):
    xVector_from = vector[:7]
    yVector_from = vector[8:15]
    xVector_to = vector[16:23]
    yVector_to = vector[24:31]
    pawnPromotionVector = vector[32:]



    return

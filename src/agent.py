import tensorflow as tf
import zipfile
import shutil
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
PAWN_PROMOTION_INDEX = [*"qrbn"]

class Agent(object):
  def __init__(self, color="w", learningRate=0.008, discount=0.5, verbose=0):
    self.init_network()
    if color == "w":
      self.color = chess.WHITE
    else:
      self.color = chess.BLACK
    self.isWhite = self.color
    self.lr = learningRate
    self.discount = discount
    self.verbose = verbose


  def init_network(self):
    # define model
    '''self.model = tf.keras.Sequential()
    
    self.model.add(tf.keras.layers.InputLayer(input_shape=(8, 8, 8), name="input_layer"))
    self.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=1, activation='relu', name="block1_conv1"))
    self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu', name="block1_conv2"))
    self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation='relu', name="block1_conv3"))
    self.model.add(tf.keras.layers.Resizing(height=128, width=128, interpolation='bilinear', crop_to_aspect_ratio=False))
    self.model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu', name="block2_conv1"))
    self.model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu', name="block2_conv2"))
    self.model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=1, activation='relu', name="block2_conv3"))
    self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    self.model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=1, activation='relu', name="block3_conv1"))
    self.model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu', name="block3_conv2"))
    self.model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu', name="block3_conv3"))
    self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5)))
    self.model.add(tf.keras.layers.Flatten())
    self.model.add(tf.keras.layers.Dense(units=2048, activation='relu'))
    #self.model.add(tf.keras.layers.Dense(units=1024, activation='relu',))
    self.model.add(tf.keras.layers.Dense(4096, activation='softmax', name="output_layer"))

    # compile model
    opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])'''
    
    model = tf.keras.Sequential()
  
    model.add(tf.keras.layers.InputLayer(input_shape=(6, 8, 8), name="input_layer"))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=1, activation='relu', name="block1_conv1"))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu', name="block1_conv2"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation='relu', name="block1_conv3"))
    model.add(tf.keras.layers.Resizing(height=64, width=64, interpolation='bilinear', crop_to_aspect_ratio=False))
    model.add(tf.keras.layers.Dropout(rate=0.05))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu', name="block2_conv1"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation='relu', name="block2_conv2"))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu', name="block2_conv3"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(8, 8)))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=2048, activation='sigmoid'))
    #model.add(tf.keras.layers.Dense(units=1024, activation='relu',))
    model.add(tf.keras.layers.Dense(4096, activation='softmax', name="output_layer"))
    
    # compile model
    opt = tf.keras.optimizers.SGD(learning_rate=0.008, momentum=0.0, weight_decay=0.0)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    self.model = model

  def freeze_model(self):
    self.frozen_model = tf.keras.models.clone_model(self.model)
    opt = tf.keras.optimizers.SGD(learning_rate=self.lr, momentum=0.0, weight_decay=0.0)
    self.frozen_model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    self.frozen_model.set_weights(self.model.get_weights())

  def find_move(self, state):
    print(np.array(state).shape)
    return np.array(self.model.predict(np.expand_dims(state, axis=0))).flatten()
  
  def update_network(self, batch, gameMemory, epochs=1):
    states, moves, rewards, new_states = [], [], [], []
    gameStates, gameMoves, gameRewards = [], [], []
    tempStates, tempMoves, tempRewards = [], [], []
    temp_diff_error = []
    for sample in batch:
      states.append(sample[0])
      moves.append(sample[1])
      rewards.append(sample[2])
      new_states.append(sample[3])
    # get information on last 10 moves and average rewards
    # to encourage bot to set up longer actions
    i = 0
    for sample in gameMemory:
      i += 1
      tempStates.append(sample[0])
      tempMoves.append(sample[1])
      tempRewards.append(sample[2])
      if i % 10 == 0:
        idx1 = (i//10)-1
        idx2 = i//10
        gameStates += tempStates[idx1:idx2]
        gameMoves += tempMoves[idx1:idx2]
        gameRewards.append(sum(tempRewards[idx1:idx2])/10)

    # The Q target
    q_target = np.array(rewards) + self.discount * np.max(self.model.predict(np.stack(new_states, axis=0), verbose=self.verbose), axis=1)

    # The Q value for the remaining actions
    q_state = self.model.predict(np.stack(states, axis=0), verbose=self.verbose)  # batch x 64 x 64

    # Combine the Q target with the other Q values.
    q_state = np.reshape(q_state, (len(batch), 64, 64))
    for idx, move in enumerate(moves):
      temp_diff_error.append(q_state[idx, move[0], move[1]] - q_target[idx])
      q_state[idx, move[0], move[1]] += q_target[idx]
    q_state = np.reshape(q_state, (len(batch), 4096))

    # Perform a step of minibatch Gradient Descent.
    self.model.fit(x=np.stack(states, axis=0), y=q_state, epochs=epochs, verbose=self.verbose)

    # for long term play memory
    # The Q target
    if len(gameStates) > 0:
      q_target = np.array(gameRewards) * 0.25

      # The Q value for the remaining actions
      q_state = self.model.predict(np.stack(gameStates, axis=0), verbose=self.verbose)  # batch x 64 x 64

      # Combine the Q target with the other Q values.
      # no need for temp_diff_error here
      q_state = np.reshape(q_state, (len(gameStates), 64, 64))
      for idx, move in enumerate(gameMoves):
        q_state[idx, move[0], move[1]] += q_target[idx//10]
      q_state = np.reshape(q_state, (len(gameStates), 4096))

      # Perform a step of minibatch Gradient Descent.
      self.model.fit(x=np.stack(gameStates, axis=0), y=q_state, epochs=epochs, verbose=self.verbose)

    return temp_diff_error





  def test_gpu(self):
      print(tf.test.gpu_device_name())

  def saveNN(self, savePath='/content/savedNNs'):
    try:
      path = os.path.join(savePath, 'nn_model')
      self.model.save(path)
      shutil.make_archive(os.path.join(savePath, 'chessNN_model'), 'zip', path)
    except Exception as e:
      print(e)

  def loadNN(self, savePath='/content/savedNNs'):
    try:
      path = os.path.join(savePath, 'nn_model')
      with zipfile.ZipFile(os.path.join(savePath, "chessNN_model.zip"), 'r') as zip_ref:
        zip_ref.extractall('/content/nn_model')
      self.model = tf.keras.models.load_model('/content/nn_model')
    except Exception as e:
      print(e)
  
  @staticmethod
  def one_hot_encode(boardState, color=chess.WHITE):
    vector = np.zeros(shape=(6,8,8))
    if color:
      for square in range(64):
        piece = str(boardState.piece_at(square))
        if piece != "None":
          vector[PIECE_INDEX_DICT[piece.lower()]][7-square//8][square%8] = int(piece.isupper()) * 2 - 1
    else:
      for sqr in range(-63, 1):
        square = abs(sqr)
        piece = str(boardState.piece_at(square))
        if piece != "None":
          vector[PIECE_INDEX_DICT[piece.lower()]][7-square//8][square%8] = int(piece.islower()) * 2 - 1
    
    #if boardState.turn == color:
      #vector[6, :, :] = 1 / boardState.fullmove_number
    #if boardState.can_claim_draw():
      #vector[7, :, :] = 1
    
    return vector
  
  @staticmethod
  def one_hot_decode(vectorIn, boardState, color=chess.WHITE):
    if color:
      vector = np.reshape(np.squeeze(vectorIn), (64,64)) # make sure that vector is a numpy array
    else:
      vector = np.reshape(np.squeeze(np.flip(vectorIn)), (64,64)) # make sure that vector is a numpy array
    while True:
      oldTiles, newTiles = np.where(vector == np.max(vector))
      oldTiles = [chess.square_name(x) for x in oldTiles]
      newTiles = [chess.square_name(x) for x in newTiles]
      moves = [oldTiles[x] + newTiles[x] for x in range(len(oldTiles))]
      for move in moves:
        try:
          movePromote = chess.Move.from_uci(move+'q')
          moveNormal = chess.Move.from_uci(move)

          if movePromote in boardState.legal_moves:
            return movePromote
          elif moveNormal in boardState.legal_moves:
            return moveNormal
        except:
          pass
      if np.max(vector) == np.min(vector) or np.max(vector) == 0:
        return False # just in case
      vector[vector == np.max(vector)] = 0 # set max to 0, then cycle back and check the next highest value for legal moves
    





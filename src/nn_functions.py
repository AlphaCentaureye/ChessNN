import tensorflow as tf
import zipfile
import os

def define_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(8,8, 13), name="input_layer"))
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
    model.add(tf.keras.layers.Dense(132, activation='softmax', name="output_layer"))
    return model

def test_gpu():
    print(tf.test.gpu_device_name())

def saveNN(model):
  path = os.path.join(os.getcwd(), '/savedNNs/nn_model')
  model.save(path)
  with zipfile.ZipFile("/savedNNs/chessNN_model.zip", 'w') as zip_ref:
    zip_ref.write("/savedNNs/chessNN_model")

def loadNN():
  path = '/savedNNs/nn_model'
  with zipfile.ZipFile("/savedNNs/chessNN_model.zip", 'r') as zip_ref:
    zip_ref.extractall()
  model = tf.keras.models.load_model(path)
  return model


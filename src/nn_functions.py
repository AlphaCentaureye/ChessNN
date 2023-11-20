import tensorflow as tf

def define_model():
    #model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(8,8, 13), name="input_layer"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation='relu', name="block1_conv1"))
    return model
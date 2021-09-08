from tensorflow import keras
class Etching_Layer(keras.layers.Layer):
    def __init__(self, num_sub, num_super):
        super(Etching_Layer, self).__init__()
        self.sub = num_sub
        self.super = num_super
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.sub, self.super),
            initializer="ones",
            trainable=True,
        )
    def call(self, inputs):
        return inputs*self.w
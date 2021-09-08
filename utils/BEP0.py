import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, ReLU, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import Input

class Back_etching_propagation():
    def __init__(self, trained_model):
        input_shape = trained_model.input.shape[1:]
        input_layer = tf.keras.Input(input_shape)
        #outputs = []
        x = input_layer
        for layer in trained_model.layers:
            print(len(layer.get_weights()), "layer name : ", layer.name)
            if len(layer.get_weights())>0:
                if layer.__class__.__name__ == 'Conv2D':
                    num_super = layer.filters
                elif layer.__class__.__name__ == 'Dense':
                    num_super = layer.units
                else:
                    raise ValueError("layer type is wrong")
                num_sub = x.shape[-1]
                x = x[...,tf.newaxis]
                x = Etching_Layer(num_sub, num_super)(x)
                layer = convert_to_separate_perceptrons(layer)
                x,o = layer(x)
                #outputs.append(o)
            else:
                x = layer(x)

        self.model = tf.keras.Model(input_layer, x)
        self.etching_layer_indice = []
        for idx, layer in enumerate(self.model.layers):
            if layer.__class__.__name__ == 'Etching_Layer':
                self.etching_layer_indice.append(idx)
    def etching(self, class_id, X):
        for is_last,idx in enumerate(self.etching_layer_indice[::-1]):
            for layer in self.model.layers:
                layer.trainable = False
            self.model.layers[idx].trainable = True #turn on the target etching layer.
            i = self.model.layers[idx].output
            _,o = self.model.layers[idx+1](i)
            num_feature = o.shape[-1]
            if is_last == 0:
                etching_model = tf.keras.Model(self.model.input, o[:,class_id])
                optimizer = tf.keras.optimizers.SGD(learning_rate=1E-3)
                for x in X:
                    with tf.GradientTape() as tape:
                        pred = etching_model(x, training=True)
                        loss = tf.reduce_mean(pred)
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                etching_weights = self.model.layers[idx].get_weights()[0]
                np.save("{}-{}.npy".format(class_id, idx), etching_weights)
            else:
                for f in range(num_feature):
                    etching_model = tf.keras.Model(self.model.input, o[:,f])
                    optimizer = tf.keras.optimizers.SGD(learning_rate=1E-3)
                    for x in X:
                        with tf.GradientTape() as tape:
                            pred = etching_model(x, training=True)
                            loss = tf.reduce_mean(pred)
                        gradients = tape.gradient(loss, self.model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    etching_weights = self.model.layers[idx].get_weights()[0]
                np.save("{}-{}.npy".format(class_id, idx), etching_weights)


    def return_model(self):
        return self.model

class Etching_Layer(tf.keras.layers.Layer):
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

def convert_convolution(conv):
    num_filters = conv.filters
    weight_shape = conv.get_weights()[0].shape[:-2]
    input_layer = tf.keras.Input(conv.input.shape[1:]+num_filters) # [B,H,W,D,N]. B:batch size, N:feature maps of previous layer
    sub_features = tf.split(input_layer, num_filters ,axis=-1) #[B,H,W,D,1] x N

    outputs = []
    for i, sub_feature in enumerate(sub_features):
        sub_feature = tf.squeeze(sub_feature, axis=-1)
        filter = Conv2D(1, weight_shape) # create a new perceptron
        out = filter(sub_feature) # pass input tensor
        filter.set_weights([conv.get_weights()[0][...,i:i+1], #set filter weight
                            conv.get_weights()[1][i:i+1]]) #set filter bias

        outputs.append(out)
    layer_output = tf.concat(outputs, axis=-1)
    o = GlobalAveragePooling2D()(layer_output)
    x = tf.keras.Model(inputs=input_layer, outputs=[layer_output, o])
    return x
def convert_dense(fc):
    num_perceptrons = fc.units
    weight_shape = fc.get_weights()[0].shape[:-2]
    input_layer = tf.keras.Input(fc.input.shape[1:]+num_perceptrons) # [B,D,N]. B:batch size, N:feature maps of previous layer
    sub_features = tf.split(input_layer, num_perceptrons ,axis=-1) #[B,D,1] x N

    outputs = []
    for i, sub_feature in enumerate(sub_features):
        sub_feature = tf.squeeze(sub_feature, axis=-1)
        perceptron = Dense(1) # create a new perceptron
        out = perceptron(sub_feature) # pass input tensor
        perceptron.set_weights([fc.get_weights()[0][...,i:i+1], #set filter weight
                                fc.get_weights()[1][i:i+1]]) #set filter bias
        outputs.append(out)
    layer_output = tf.concat(outputs, axis=-1)
    o = layer_output
    x = tf.keras.Model(inputs=input_layer, outputs=[layer_output, o])
    return x

def convert_to_separate_perceptrons(layer):
    layer_type = layer.__class__.__name__
    print(layer_type)
    if layer_type == 'Conv2D':
        converted = convert_convolution(layer)
    elif layer_type == 'Dense':
        converted = convert_dense(layer)
    else:
        raise NotImplementedError("This type of layer is not implemented yet.")
    return converted

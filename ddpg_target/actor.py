import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, BatchNormalization, GaussianNoise, Flatten, concatenate
from keras.layers import Conv2D, MaxPool2D, Dropout

class Actor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        #self.act_range = act_range
        self.tau = tau
        self.lr = lr
        self.model = self.network()
        self.target_model = self.network(); self.target_model.set_weights(self.model.get_weights())
        self.adam_optimizer = self.optimizer()

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        state_input = Input(shape= [self.env_dim], name='state_input')
        X = Dense(128, activation = "relu")(state_input)
        X = Dense(128, activation='relu')(X)
        Out = Dense(self.act_dim, activation='sigmoid', kernel_initializer=RandomUniform())(X)
        #continuous action
        #out = Dense(self.act_dim, activation='tanh', kernel_initializer=RandomUniform())(X)
        #out = Lambda(lambda i: i * self.act_range)(out)
        #
        return Model(state_input, Out)

    def predict(self, state):
        """ Action prediction
        """
        return self.model.predict(np.expand_dims(state, axis=0))

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model.predict(inp)


    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, states, actions, grads):
        """ Actor Training
        """
        self.adam_optimizer([states, grads])

    def optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.act_dim))
        temp = K.placeholder(shape=(None, 1))
        # action_gdts = K.placeholder(shape= [self.env_dim])
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function(inputs=[self.model.input, action_gdts], outputs=[ K.constant(1)],updates=[tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])
        

    def save(self, filename):
        try:
            self.model.save_weights(filename)
        except:
            print("ERROR! Save Paras File unsuccess: " + filename)
        else:
            print("Save Paras File success: " + filename)

    def load_weights(self, filename):
        try:
            self.model.load_weights(filename)
        except:
            print("ERROR! Load Paras File unsuccess: " + filename)
        else:
            print("Load Paras File success: " + filename)
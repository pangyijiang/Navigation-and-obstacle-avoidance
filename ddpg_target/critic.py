import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten, Add, ReLU
from keras.layers import Conv2D, MaxPool2D, Dropout

class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        # Build models and target models
        self.model = self.network()
        self.target_model = self.network(); self.target_model.set_weights(self.model.get_weights())
        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        action_input = Input(shape = [self.act_dim])
        X_2 = Dense(64, activation='linear')(action_input)
        # X_2 = Dense(64, activation='relu')(X_2)

        state_input = Input(shape= [self.env_dim])
        X_1 = Dense(64, activation='linear')(state_input)
        # X_1 = Dense(64, activation='relu')(X_1)

        # X = concatenate([X_1, X_2])
        X= Add()([X_1, X_2]) 
        X = ReLU()(X)
        X = Dense(64, activation='relu')(X)
        # x = Dense(32, activation='relu')(X)
        out = Dense(1, activation='linear')(X)

        model = Model([state_input, action_input], out)
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        return self.action_grads([states, actions])

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        return self.model.train_on_batch([states, actions], critic_target)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

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
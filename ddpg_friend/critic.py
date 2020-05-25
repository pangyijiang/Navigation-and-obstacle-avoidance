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
        # state_input_0 = Input(shape= self.env_dim[0])
        # X_0 = Dense(256, activation = "relu")(state_input_0)
        # X_0 = Dropout(rate=0.3)(X_0)
        # X_0 = Dense(128, activation='relu')(X_0)

        action_input = Input(shape = [self.act_dim])
        X_2 = Dense(256, activation='relu')(action_input)
        X_2 = Dropout(rate=0.3)(X_2)
        X_2 = Dense(128, activation='relu')(X_2)

        state_input = Input(shape= self.env_dim)
        X_1 = Conv2D(filters=4, kernel_size=(3, 3), padding='SAME', activation='relu')(state_input)
        X_1= Conv2D(filters=4, kernel_size=(3, 3), padding='SAME', activation='relu')(X_1)
        X_1= BatchNormalization(momentum=0.15)(X_1)
        X_1= MaxPool2D(pool_size=(2, 2))(X_1)
        X_1= Conv2D(filters=4, kernel_size=(5, 5), padding='SAME', activation='relu')(X_1)
        X_1= BatchNormalization(momentum=0.15)(X_1)
        X_1= Dropout(rate=0.3)(X_1)

        X_1= Conv2D(filters=8, kernel_size=(3, 3), padding='SAME', activation='relu')(X_1)
        X_1= Conv2D(filters=8, kernel_size=(3, 3), padding='SAME', activation='relu')(X_1)
        X_1= BatchNormalization(momentum=0.15)(X_1)
        X_1= MaxPool2D(pool_size=(2, 2))(X_1)
        X_1= Conv2D(filters=8, kernel_size=(5, 5), padding='SAME', activation='relu')(X_1)
        X_1= BatchNormalization(momentum=0.15)(X_1)
        X_1= Dropout(rate=0.3)(X_1)

        X_1= Conv2D(filters=16, kernel_size=(3, 3), padding='SAME', activation='relu')(X_1)
        X_1= Conv2D(filters=16, kernel_size=(3, 3), padding='SAME', activation='relu')(X_1)
        X_1= BatchNormalization(momentum=0.15)(X_1)
        X_1= MaxPool2D(pool_size=(2, 2))(X_1)
        X_1= Conv2D(filters=16, kernel_size=(5, 5), padding='SAME', activation='relu')(X_1)
        X_1= BatchNormalization(momentum=0.15)(X_1)
        X_1= Dropout(rate=0.3)(X_1)

        X_1= Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(X_1)
        X_1= Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(X_1)
        X_1= BatchNormalization(momentum=0.15)(X_1)
        X_1= MaxPool2D(pool_size=(2, 2))(X_1)
        X_1= Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(X_1)
        X_1= BatchNormalization(momentum=0.15)(X_1)
        X_1= Dropout(rate=0.3)(X_1)

        X_1= Flatten()(X_1)
        X_1= Dense(256, activation = "relu")(X_1)
        X_1= Dropout(rate=0.3)(X_1)
        X_1= Dense(128, activation='relu')(X_1)

        

        X = concatenate([X_1, X_2])
        # X= Add()([X_1, X_2]) 
        # X = ReLU()(X)
        X = Dense(128, activation='relu')(X)
        x = Dense(64, activation='relu')(X)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(X)

        model = Model([state_input, action_input], out)
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        # batch_state_1, batch_state_2= [],[]
        # for i in states:
        #     batch_state_1.append(i[0])
        #     batch_state_2.append(i[1])
        # batch_state_1 = np.array(batch_state_1)
        # batch_state_2 = np.array(batch_state_2)

        return self.action_grads([states, actions])

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        # batch_state_1, batch_state_2= [],[]
        # for i in inp[0]:
        #     batch_state_1.append(i[0])
        #     batch_state_2.append(i[1])
        # batch_state_1 = np.array(batch_state_1)
        # batch_state_2 = np.array(batch_state_2)
        # return self.target_model.predict([batch_state_1, batch_state_2, inp[1]])
        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        # batch_state_1, batch_state_2= [],[]
        # for i in states:
        #     batch_state_1.append(i[0])
        #     batch_state_2.append(i[1])
        # batch_state_1 = np.array(batch_state_1)
        # batch_state_2 = np.array(batch_state_2)

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
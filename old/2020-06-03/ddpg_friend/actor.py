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
        # state_input_0 = Input(shape= self.env_dim[0], name='state_input_0')
        # X_0 = Dense(256, activation = "relu")(state_input_0)
        # X_0 = Dropout(rate=0.3)(X_0)
        # X_0 = Dense(128, activation='relu')(X_0)


        state_input = Input(shape= self.env_dim, name='state_input')
        #inp = Input((self.env_dim))
        #
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
        X_1= Dropout(rate=0.3)(X_1)
        X_1= Dense(64, activation='relu')(X_1)

        # X= Add()([X_1, X_2]) 
        # X = ReLU()(X)
        #X = Dense(128, activation='relu')(X_1)
        # x = Dense(64, activation='relu')(X)
        #discrete action
        Out = Dense(self.act_dim, activation='sigmoid', kernel_initializer=RandomUniform())(X_1)
        #continuous action
        #out = Dense(self.act_dim, activation='tanh', kernel_initializer=RandomUniform())(X)
        #out = Lambda(lambda i: i * self.act_range)(out)
        #
        return Model(state_input, Out)

    def predict(self, state):
        """ Action prediction
        """
        return self.model.predict(np.expand_dims(state, axis=0))
        # return self.model.predict(state)
        # return self.model.predict([state[0][np.newaxis,:], state[1][np.newaxis, :, :, :]])

    # def batch_predict(self, state):
    #     """ Action prediction
    #     """
    #     # batch_state_1, batch_state_2= [],[]
    #     # for i in state:
    #     #     batch_state_1.append(i[0])
    #     #     batch_state_2.append(i[1])
    #     # batch_state_1 = np.array(batch_state_1)
    #     # batch_state_2 = np.array(batch_state_2)
    #     # return self.model.predict(np.expand_dims(state, axis=0))
    #     # return self.model.predict([batch_state_1, batch_state_2])
    #     return self.model.predict(np.expand_dims(state, axis=0))

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        # batch_state_1, batch_state_2= [],[]
        # for i in inp:
        #     batch_state_1.append(i[0])
        #     batch_state_2.append(i[1])
        # batch_state_1 = np.array(batch_state_1)
        # batch_state_2 = np.array(batch_state_2)

        # return self.target_model.predict([batch_state_1, batch_state_2])
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
        # batch_state_1, batch_state_2= [],[]
        # for i in states:
        #     batch_state_1.append(i[0])
        #     batch_state_2.append(i[1])
        # batch_state_1 = np.array(batch_state_1)
        # batch_state_2 = np.array(batch_state_2)

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
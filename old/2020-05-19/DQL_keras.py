import random
import numpy as np
from collections import deque
from keras.layers import Dense, Input, BatchNormalization, Conv2D, Flatten, LeakyReLU, Concatenate, Activation
from keras.layers.core import Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import keras

class DQN_keras:
    batch_size = 512
    exploration_rate = 1.0  # exploration rate
    tau = 0.1   #model_target update rate
    def __init__(self, state_size, action_size, gamma = 0.9, exploration_rate_min = 0.01, learning_rate = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 10000)
        self.gamma = gamma    # discount rate
        self.exploration_rate_min = exploration_rate_min
        self.learning_rate = learning_rate
        self.model = self._build_model_target()
        self.model_target = self._build_model_target();self.model_target.set_weights(self.model.get_weights())

    def _build_model(self):

        input_1 = Input(shape = [self.state_size[0]])
        X_1 = Dense(128, kernel_initializer='glorot_uniform')(input_1)
        X_1 = LeakyReLU(0.2)(X_1)
        X_1 = Dropout(0.3)(X_1)
        X_1 = Dense(64, kernel_initializer='glorot_uniform')(X_1)
        X_1 = LeakyReLU(0.2)(X_1)
        X_1 = Dropout(0.3)(X_1)
        X_1 = Dense(64, kernel_initializer='glorot_uniform')(X_1)
        X_1 = LeakyReLU(0.2)(X_1)
        X_1 = Dropout(0.3)(X_1)
        X_1 = Dense(32, kernel_initializer='glorot_uniform')(X_1)
        X_1 = LeakyReLU(0.2)(X_1)
        X_1 = Dropout(0.3)(X_1)
        X_1 = Dense(32, kernel_initializer='glorot_uniform')(X_1)
        X_1 = LeakyReLU(0.2)(X_1)
        X_1 = Dropout(0.3)(X_1)

        input_2 = Input(shape = self.state_size[1])
        X_2 = Conv2D(filters = 4 , kernel_size = (3, 3) , strides=(2, 2), padding = 'same', kernel_initializer='glorot_uniform')(input_2)
        X_2 = LeakyReLU(0.2)(X_2)
        X_2 = Dropout(0.3)(X_2)
        X_2 = Conv2D(filters = 8 , kernel_size = (3, 3) , strides=(2, 2), padding = 'same', kernel_initializer='glorot_uniform')(X_2)
        X_2 = LeakyReLU(0.2)(X_2)
        X_2 = Dropout(0.3)(X_2)
        X_2 = Conv2D(filters = 16 , kernel_size = (3, 3) , strides=(2, 2), padding = 'same', kernel_initializer='glorot_uniform')(X_2)
        X_2 = LeakyReLU(0.2)(X_2)
        X_2 = Dropout(0.3)(X_2)
        X_2 = Conv2D(filters = 32 , kernel_size = (3, 3) , strides=(2, 2), padding = 'same', kernel_initializer='glorot_uniform')(X_2)
        X_2 = LeakyReLU(0.2)(X_2)
        X_2 = Dropout(0.3)(X_2)
        X_2 = Conv2D(filters = 64 , kernel_size = (3, 3) , strides=(2, 2), padding = 'same', kernel_initializer='glorot_uniform')(X_2)
        X_2 = LeakyReLU(0.2)(X_2)
        X_2 = Dropout(0.3)(X_2)
        X_2 = Flatten()(X_2)

        X = Concatenate()([X_1, X_2])
        X = Dense(256, kernel_initializer='glorot_uniform')(X)
        X = LeakyReLU(0.2)(X)
        X = Dropout(0.3)(X)
        X = Dense(128, kernel_initializer='glorot_uniform')(X)
        X = LeakyReLU(0.2)(X)
        X = Dropout(0.3)(X)
        X = Dense(64, kernel_initializer='glorot_uniform')(X)
        X = LeakyReLU(0.2)(X)
        X = Dropout(0.3)(X)
        Out = Dense(self.action_size, activation='linear', kernel_initializer='glorot_uniform')(X)

        model =  Model(inputs=[input_1, input_2], outputs=[Out])
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #print(model.summary())
        return model
    def _build_model_target(self):
        input_1 = Input(shape = [self.state_size[0]])
        X_1 = Dense(512, kernel_initializer='glorot_uniform')(input_1)
        X_1 = Activation('relu')(X_1)
        X_1 = Dense(256, kernel_initializer='glorot_uniform')(X_1)
        X_1 = Activation('relu')(X_1)
        X_1 = Dense(256, kernel_initializer='glorot_uniform')(X_1)
        X_1 = Activation('relu')(X_1)

        Out = Dense(self.action_size, kernel_initializer='glorot_uniform')(X_1)
        model =  Model(inputs= input_1, outputs = Out)
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_model_obstacle(self):
        pass
 
    def _build_model_friend(self):
        pass

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, flag_train):
        if np.random.rand() <= self.exploration_rate and flag_train:
            action =  random.randrange(self.action_size)
        else:
            # act_values = self.model.predict([state[0][np.newaxis, :], state[1][np.newaxis, :, :, :]])
            act_values = self.model.predict(state[0][np.newaxis, :])
            action = np.argmax(act_values[0])
        return action  # returns action

    def replay_minibatch(self, episode = 5):
        for i in range(episode):
            loss = []
            if len(self.memory) > self.batch_size:
                minibatch = random.sample(self.memory, self.batch_size)
                targets_f_batch = []
                states_1_batch, states_2_batch = [], []
                for state, action, reward, next_state, done in minibatch:
                    # target_f = self.model.predict([state[0][np.newaxis, :], state[1][np.newaxis, :, :, :]]) #target: Q_a1, Q_a2, Q_a3...
                    target_f = self.model.predict(state[0][np.newaxis, :]) #target: Q_a1, Q_a2, Q_a3...
                    
                    if done:
                        target_f[0][action] = reward
                    else:
                        # target_f[0][action] = reward + self.gamma * np.amax(self.model_target.predict([state[0][np.newaxis, :], state[1][np.newaxis, :, :, :]])[0]) 
                        target_f[0][action] = reward + self.gamma * np.amax(self.model_target.predict(state[0][np.newaxis, :])[0]) 
                    
                    states_1_batch.append(state[0])
                    states_2_batch.append(state[1])
                    targets_f_batch.append(target_f[0])

                states_1_batch = np.array(states_1_batch)
                states_2_batch = np.array(states_2_batch)
                # history = self.model.fit([states_1_batch, states_2_batch], np.array(targets_f_batch), epochs=1, verbose=0)  #fit, update para based on env,action_Q 
                history = self.model.fit(states_1_batch, np.array(targets_f_batch), epochs=1, verbose=0)  #fit, update para based on env,action_Q 
                loss.append(history.history['loss'][0])
        return np.average(loss)   #loss
    
    # def replay(self, batch_size):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target_f = self.model.predict(state) #target: Q_a1, Q_a2, Q_a3...
    #         if done:
    #             target_f[0][action] = reward
    #         if not done:
    #             target_f[0][action] = reward + self.gamma * np.amax(self.model_target.predict(next_state)[0]) 
    #         history = self.model.fit(state, target_f, epochs=1, verbose=0)  #fit, update para based on env,action_Q 

    def update_model_target(self):
        # para = self.model.get_weights()
        # para_target = self.target_model.get_weights()
        # for i in range(len(para)):
        #     para_target[i] = self.tau * para[i] + (1 - self.tau) * para_target[i]
        # self.target_model.set_weights(para_target)
        self.model_target.set_weights(self.model.get_weights())

    def exploration_rate_decay(self, rate_decay = 0.99):
        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= rate_decay

    def load(self, name):
        self.model.load_weights(name)
        #self.model.load_model(name)
        print("Load model weight success")

    def save(self, name):
        self.model.save(name)    
        print("Save model weight success")

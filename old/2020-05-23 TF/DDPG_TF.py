import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from collections import deque
from PIL import Image
import numpy as np
import time
import random

class DDPG(object):
    MEMORY_CAPACITY = 2000
    TAU = 0.01      # soft replacement
    LearnRate_Actor = 0.001    # learning rate for actor
    LearnRate_Critic = 0.002    # learning rate for critic
    GAMMA = 0.9     # reward discount
    BATCH_SIZE = 32
    """
    input: 
        a_bound: limit of value range.
    """
    
    def __init__(self, a_dim, s_1_dim, s_2_dim = (128,128,1), epsilon_min = 0.01):
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.9999
        # self.memory = np.zeros((self.MEMORY_CAPACITY, s_1_dim * 2 + a_dim + 1), dtype=np.float32)
        self.memory = deque(maxlen = self.MEMORY_CAPACITY)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_1_dim, self.s_2_dim = a_dim, s_1_dim, s_2_dim
        self.state_1 = tf.placeholder(tf.float32, [None, s_1_dim], 's')
        self.state_2 = tf.placeholder(tf.float32, [None, 128,128,1], 's')
        self.next_state_1 = tf.placeholder(tf.float32, [None, s_1_dim], 's_')
        self.next_state_2 = tf.placeholder(tf.float32, [None, 128,128,1], 's_')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'r')

        
        self.actor = self._build_actor_NN(self.state_1, self.state_2)
        self.critic = self._build_critic_NN(self.state_1, self.state_2, self.actor)
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        actor_target = self._build_actor_NN(self.next_state_1, self.state_2, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        critic_target = self._build_critic_NN(self.next_state_1, self.state_2, actor_target, reuse=True, custom_getter=ema_getter)

        cost_function_actor = - tf.reduce_mean(self.critic)  # maximize the q
        self.Optimizer_actor = tf.train.AdamOptimizer(self.LearnRate_Actor).minimize(cost_function_actor, var_list=a_params)

        #tf.control_dependencies(target_update): "target_update" must be executed before running the operations defined in the context
        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.reward + self.GAMMA * critic_target
            cost_function_critic = tf.losses.mean_squared_error(labels=q_target, predictions = self.critic) #td_error
            self.Optimizer_critic = tf.train.AdamOptimizer(self.LearnRate_Critic).minimize(cost_function_critic, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, s, flag_train):
        act_values =  self.sess.run(self.actor, {self.state_1: s[0][np.newaxis, :], self.state_2: s[1][np.newaxis, :]})[0]
        if np.random.rand() <= self.epsilon and flag_train:
            p = np.random.randint(0 , self.a_dim)
            act_values[p] = 1.0
        action = np.argmax(act_values)
        return act_values, action
        
    def learn(self):
        # indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        # bt = self.memory[indices, :]
        # batch_state_1 = bt[:, :self.s_1_dim]

        minibatch = random.sample(self.memory, self.BATCH_SIZE)
        batch_state_1, batch_state_2, batch_action, batch_reward, batch_next_state_1, batch_next_state_2 = [],[],[],[],[],[]
        for i in minibatch:
            batch_state_1.append(i[0])
            batch_state_2.append(i[1])
            batch_action.append(i[2])
            batch_reward.append([i[3]])
            batch_next_state_1.append(i[4])
            batch_next_state_2.append(i[5])
        batch_state_1 = np.array(batch_state_1)
        batch_state_2 = np.array(batch_state_2)
        batch_action = np.array(batch_action)
        batch_reward = np.array(batch_reward)
        batch_next_state_1 = np.array(batch_next_state_1)
        batch_next_state_2 = np.array(batch_next_state_2)
        #check obs
        # flag = False
        # if flag:
        #     temp =batch_state_2[0]*255 + 127
        #     temp = temp.astype("int")
        #     temp = temp.reshape((128,128))
        #     import matplotlib.pyplot as plt
        #     plt.imshow(temp, cmap='gray')
        #     plt.title("train_labels" )
        #     plt.show()
        self.sess.run(self.Optimizer_actor, {self.state_1: batch_state_1, self.state_2: batch_state_2})
        self.sess.run(self.Optimizer_critic, {self.state_1: batch_state_1, self.state_2: batch_state_2, self.actor: batch_action, self.reward: batch_reward, self.next_state_1: batch_next_state_1, self.next_state_2: batch_next_state_2})
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, s, a, r, s_):
        # index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory.append([s[0], s[1], a, r, s_[0], s_[1]])
        # self.pointer += 1
        # transition = np.hstack((s[0], s[1], a, [r], s_[0], s_[1]))
        # index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        # self.pointer += 1

    def _build_actor_NN(self, state_1, state_2, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            X_1 = tf.layers.dense(inputs = state_1, units = 256, activation='relu', trainable=trainable)
            X_1 = tf.layers.dense(inputs = X_1, units = 128, activation='relu', trainable=trainable)

            X_2 = tf.layers.Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', activation='relu', input_shape = self.s_2_dim, trainable=trainable)(state_2)
            X_2 = tf.layers.Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', activation='relu', trainable=trainable)(X_2)
            # X_2 = tf.layers.BatchNormalization(momentum=0.15, trainable=trainable)(X_2)
            X_2 = tf.nn.max_pool2d(input = X_2, ksize = 2, strides = 2, padding = "SAME")
            X_2 = tf.layers.Conv2D(filters=8, kernel_size=(5, 5), padding='SAME', activation='relu')(X_2)
            X_2 = tf.nn.dropout(X_2, 0.3)

            X_2 = tf.layers.Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same', activation='relu', trainable=trainable)(X_2)
            X_2 = tf.layers.Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same', activation='relu', trainable=trainable)(X_2)
            # X_2 = tf.layers.BatchNormalization(momentum=0.15, trainable=trainable)(X_2)
            X_2 = tf.nn.max_pool2d(input = X_2, ksize = 2, strides = 2, padding = "SAME")
            X_2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='SAME', activation='relu', trainable=trainable)(X_2)
            X_2 = tf.nn.dropout(X_2, 0.3)

            X_2 = tf.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation='relu', trainable=trainable)(X_2)
            X_2 = tf.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation='relu', trainable=trainable)(X_2)
            # X_2 = tf.layers.BatchNormalization(momentum=0.15, trainable=trainable)(X_2)
            X_2 = tf.nn.max_pool2d(input = X_2, ksize = 2, strides = 2, padding = "SAME")
            X_2 = tf.layers.Conv2D(filters = 32, kernel_size=(5, 5), padding='SAME', activation='relu')(X_2)
            X_2 = tf.nn.dropout(X_2, 0.3)

            X_2 = tf.layers.Flatten()(X_2)
            X_2 = tf.layers.Dense(256, activation='relu', trainable=trainable)(X_2)
            X_2 = tf.nn.dropout(X_2, 0.3)
            X_2 = tf.layers.Dense(128, activation='relu', trainable=trainable)(X_2)

            X = tf.concat([X_1, X_2], axis = 1)
            # X= tf.math.add_n([X_1, X_2])
            X = tf.layers.Dense(256, activation='relu', trainable=trainable)(X)
            X = tf.layers.Dense(128, activation='relu', trainable=trainable)(X)

            actor = tf.layers.dense(inputs = X, units = self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return actor   

    def _build_critic_NN(self, state_1, state_2, actor, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            X_1 = tf.layers.dense(inputs = state_1, units = 256, activation='relu', trainable=trainable)
            X_1 = tf.layers.dense(inputs = X_1, units = 128, activation='relu', trainable=trainable)

            X_2 = tf.layers.Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', activation='relu', input_shape = self.s_2_dim, trainable=trainable)(state_2)
            X_2 = tf.layers.Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', activation='relu', trainable=trainable)(X_2)
            # X_2 = tf.layers.BatchNormalization(momentum=0.15, trainable=trainable)(X_2)
            X_2 = tf.nn.max_pool2d(input = X_2, ksize = 2, strides = 2, padding = "SAME")
            X_2 = tf.layers.Conv2D(filters=8, kernel_size=(5, 5), padding='SAME', activation='relu')(X_2)
            X_2 = tf.nn.dropout(X_2, 0.3)

            X_2 = tf.layers.Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same', activation='relu', trainable=trainable)(X_2)
            X_2 = tf.layers.Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same', activation='relu', trainable=trainable)(X_2)
            # X_2 = tf.layers.BatchNormalization(momentum=0.15, trainable=trainable)(X_2)
            X_2 = tf.nn.max_pool2d(input = X_2, ksize = 2, strides = 2, padding = "SAME")
            X_2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='SAME', activation='relu')(X_2)
            X_2 = tf.nn.dropout(X_2, 0.3)

            X_2 = tf.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation='relu', trainable=trainable)(X_2)
            X_2 = tf.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation='relu', trainable=trainable)(X_2)
            # X_2 = tf.layers.BatchNormalization(momentum=0.15, trainable=trainable)(X_2)
            X_2 = tf.nn.max_pool2d(input = X_2, ksize = 2, strides = 2, padding = "SAME")
            X_2 = tf.layers.Conv2D(filters = 32, kernel_size=(5, 5), padding='SAME', activation='relu')(X_2)
            X_2 = tf.nn.dropout(X_2, 0.3)

            X_2 = tf.layers.Flatten()(X_2)
            X_2 = tf.layers.Dense(256, activation='relu', trainable=trainable)(X_2)
            X_2 = tf.nn.dropout(X_2, 0.3)
            X_2 = tf.layers.Dense(128, activation='relu', trainable=trainable)(X_2)
            
            X_3 = tf.layers.dense(inputs = actor, units = 256, trainable=trainable)
            X_3 = tf.layers.Dense(128, activation='relu', trainable=trainable)(X_3)

            X = tf.concat([X_1, X_2, X_3], axis = 1)
            # X= tf.math.add_n([X_1, X_2, X_3])

            X = tf.layers.Dense(256, activation='relu', trainable=trainable)(X)
            X = tf.layers.Dense(128, activation='relu', trainable=trainable)(X)

            critic = tf.layers.dense(inputs = X, units = 1, trainable=trainable)
            return  critic  # Q(s,a)

    def Save_para(self, filename = 'MyModel'):
        path = './paras/'
        localtime= time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        flag = self.saver.save(self.sess, path + filename)
        if (flag != None):
            print("Save Paras File success: " + path + filename)

    def Load_para(self, filename = 'MyModel'):
        path = './paras/'
        flag = self.saver.restore(self.sess, path + filename)
        if (flag != None):
            print("Load Paras File success: " + path + filename)

            



import tensorflow as tf
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
    
    def __init__(self, a_dim, s_dim, epsilon_min = 0.01):
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.9999
        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.state = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.next_state = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'r')

        
        self.actor = self._build_actor_NN(self.state,)
        self.critic = self._build_critic_NN(self.state, self.actor, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        actor_target = self._build_actor_NN(self.next_state, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        critic_target = self._build_critic_NN(self.next_state, actor_target, reuse=True, custom_getter=ema_getter)

        cost_function_actor = - tf.reduce_mean(self.critic)  # maximize the q
        self.Optimizer_actor = tf.train.AdamOptimizer(self.LearnRate_Actor).minimize(cost_function_actor, var_list=a_params)

        #tf.control_dependencies(target_update): "target_update" must be executed before running the operations defined in the context
        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.reward + self.GAMMA * critic_target
            cost_function_critic = tf.losses.mean_squared_error(labels=q_target, predictions = self.critic) #td_error
            self.Optimizer_critic = tf.train.AdamOptimizer(self.LearnRate_Critic).minimize(cost_function_critic, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def step(self, s, flag_train):
        act_values =  self.sess.run(self.actor, {self.state: s[np.newaxis, :]})[0]
        if np.random.rand() <= self.epsilon and flag_train:
            p = np.random.randint(0 , self.a_dim)
            act_values[p] = 1.0
        action = np.argmax(act_values)
        return act_values, action
        
    def learn(self):
        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        batch_state = bt[:, :self.s_dim]
        batch_action = bt[:, self.s_dim: self.s_dim + self.a_dim]
        batch_reward = bt[:, -self.s_dim - 1: -self.s_dim]
        batch_next_state = bt[:, -self.s_dim:]

        self.sess.run(self.Optimizer_actor, {self.state: batch_state})
        self.sess.run(self.Optimizer_critic, {self.state: batch_state, self.actor: batch_action, self.reward: batch_reward, self.next_state: batch_next_state})
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_actor_NN(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 128
            n_l2 = 128
            net_1 = tf.layers.dense(s, n_l1, activation=tf.nn.relu, name='l1', trainable=trainable)
            net_2 = tf.layers.dense(net_1, n_l2, activation=tf.nn.relu, name='l2', trainable=trainable)
            #a = tf.layers.dense(net_2, self.a_dim, activation=tf.keras.activations.linear, name='a', trainable=trainable)
            actor = tf.layers.dense(net_2, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            #a_bound = tf.multiply(a, self.a_bound, name='scaled_a') #Returns x * y, supports broadcasting, for example: (-1,1)*0.5 -> (-0.5,0.5)
            return actor   

    def _build_critic_NN(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 64
            n_l2 = 64
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_2 = tf.layers.dense(net_1, n_l2, activation=tf.nn.relu, name='l2', trainable=trainable)
            critic = tf.layers.dense(net_2, 1, trainable=trainable)
            # critic = tf.layers.dense(net_1, 1, trainable=trainable)
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

            



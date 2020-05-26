import sys
import numpy as np

from tqdm import tqdm
from .actor import Actor
from .critic import Critic
from memory_buffer import MemoryBuffer

class DDPG:
    batch_size = 32
    MEMORY_CAPACITY = 20000
    epsilon = 1.0  # exploration rate
    epsilon_min = 0.1
    epsilon_decay = 0.99995

    def __init__(self, a_dim, s_dim, gamma = 0.9, lr = 0.0005, tau = 0.01):
        """ Initialization
        """
        # Environment and A2C parameters
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.actor = Actor(self.s_dim, a_dim, 0.5 * lr, tau)
        self.critic = Critic(self.s_dim, a_dim, lr, tau)
        self.buffer = MemoryBuffer(self.MEMORY_CAPACITY)

    def policy_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s)[0]

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def remember(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.remember(state, action, reward, done, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        loss_critic = self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)
        grads_action = np.array(grads).reshape((-1, self.a_dim))
        # Train actor
        loss_actor = self.actor.train(states, actions, grads_action)
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()
        return loss_critic, loss_actor

    def step(self, old_state, flag_train):
        act_values = self.policy_action(old_state)
        if np.random.rand() <= self.epsilon and flag_train:
            p = np.random.randint(0 , self.a_dim)
            act_values[p] = 1.0
        action = np.argmax(act_values)
        return act_values, action

    def learn(self):
        # Sample experience from buffer
        states, actions, rewards, dones, new_states, _ = self.sample_batch(self.batch_size)
        # Predict target q-values using target networks
        q_values = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])
        # Compute critic target
        critic_target = self.bellman(rewards, q_values, dones)
        # Train both networks on sampled batch, update target networks
        loss_critic, loss_actor = self.update_models(states, actions, critic_target)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        self._epsilon_decay()
        return loss_critic, loss_actor

    def _epsilon_decay(self):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def save_weights(self, filename = 'model'):
        path = './paras/'
        name_actor = '_actor.h5'
        name_critic = '_critic.h5'
        self.actor.save(path + filename + name_actor)
        self.critic.save(path + filename + name_critic)

    def load_weights(self, filename = 'model'):
        path = './paras/'
        name_actor = '_actor.h5'
        name_critic = '_critic.h5'
        self.critic.load_weights(path + filename + name_critic)
        self.actor.load_weights(path + filename + name_actor)


class OrnsteinUhlenbeckProcess(object):
    """ Ornstein-Uhlenbeck Noise (original code by @slowbull)
    """
    def __init__(self, theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x0 = x
        return x
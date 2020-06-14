import random
import numpy as np

from collections import deque

class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size, with_per = False):
        """ Initialization
        """
        if(with_per):
            # Prioritized Experience Replay
            self.alpha = 0.5
            self.epsilon = 0.01
            self.buffer = SumTree(buffer_size)
        else:
            # Standard Buffer
            self.buffer = deque()
        self.count = 0
        self.with_per = with_per
        self.buffer_size = buffer_size

    def remember(self, state_1, action, reward, done, new_state_1, state_2=None, new_state_2=None, error=None):
        """ Save an experience to memory, optionally with its TD-Error
        """
        if state_2 == None or new_state_2 == None:
            experience = (state_1, action, reward, done, new_state_1)
        else:
            experience = (state_1, action, reward, done, new_state_1, state_2, new_state_2)
        if(self.with_per):
            priority = self.priority(error[0])
            self.buffer.add(priority, experience)
            self.count += 1
        else:
            # Check if buffer is already full
            if self.count < self.buffer_size:
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)

    def priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        return (error + self.epsilon) ** self.alpha

    def size(self):
        """ Current Buffer Occupation
        """
        return self.count

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []

        # Sample using prorities
        if(self.with_per):
            T = self.buffer.total() // batch_size
            for i in range(batch_size):
                a, b = T * i, T * (i + 1)
                s = random.uniform(a, b)
                idx, error, data = self.buffer.get(s)
                batch.append((*data, idx))
            idx = np.array([i[5] for i in batch])
        # Sample randomly from Buffer
        elif self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        if len(batch[0]) == 5:
            s_batch = np.array([i[0] for i in batch])
            a_batch = np.array([i[1] for i in batch])
            r_batch = np.array([i[2] for i in batch])
            d_batch = np.array([i[3] for i in batch])
            new_s_batch = np.array([i[4] for i in batch])
            return s_batch, a_batch, r_batch, d_batch, new_s_batch, idx
        elif len(batch[0]) == 7:
            s_1_batch = np.array([i[0] for i in batch])
            a_batch = np.array([i[1] for i in batch])
            r_batch = np.array([i[2] for i in batch])
            d_batch = np.array([i[3] for i in batch])
            new_s_1_batch = np.array([i[4] for i in batch])
            s_2_batch = np.array([i[5] for i in batch])
            new_s_2_batch = np.array([i[6] for i in batch])
            return s_1_batch, a_batch, r_batch, d_batch, new_s_1_batch, s_2_batch, new_s_2_batch, idx

    def update(self, idx, new_error):
        """ Update priority for idx (PER)
        """
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        if(self.with_per): self.buffer = SumTree(buffer_size)
        else: self.buffer = deque()
        self.count = 0

""" Original Code by @jaara: https://github.com/jaara/AI-blog/blob/master/SumTree.py
"""

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
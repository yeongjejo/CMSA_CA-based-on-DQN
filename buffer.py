import torch

import collections
import random

class ReplayBuffer():
    def __init__(self):
        self.buffer_limit = 10000
        self.batch_size = 32
        self.buffer = collections.deque(maxlen=self.buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)
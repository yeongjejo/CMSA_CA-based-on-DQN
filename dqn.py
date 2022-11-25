import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

# Hyperparameters
learning_rate = 0.0005
gamma = 0.99

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 7)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def get_action(self, obs, epsilon):
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 6)
        else:
            return self.forward(obs).argmax().item()

    def train_net(self, q, q_target, memory):
        for i in range(10):
            s, a, r, s_prime = memory.sample()

            q_out = q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime
            loss = F.smooth_l1_loss(q_a, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
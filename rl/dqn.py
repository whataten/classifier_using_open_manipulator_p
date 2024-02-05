import collections
import random
from GraspEnv import GraspEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 200
batch_size = 16

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        
        return torch.tensor(np.array(s_lst), dtype=torch.float).unsqueeze(dim=1).cuda(), torch.tensor(np.array(a_lst)).cuda(), \
               torch.tensor(np.array(r_lst)).cuda(), torch.tensor(np.array(s_prime_lst), dtype=torch.float).unsqueeze(dim=1).cuda(), \
               torch.tensor(np.array(done_mask_lst)).cuda()

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):

    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(135424, 2116)
        self.fc2 = nn.Linear(2116, 9)

    def forward(self, x):
        x = x / 255.0  # Normalize input (assuming input range [0, 255])

        x = torch.relu(self.conv1(x))

        x = torch.relu(self.conv2(x))

        x = torch.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = torch.relu(self.fc1(x))

        x = self.fc2(x)

        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        optimizer.zero_grad()
        q_out = q(s)

        a = a.unsqueeze(1)

        q_a = q_out.gather(1, a)

        max_q_prime = q_target(s_prime).max(1)[0]

        done_mask = done_mask.squeeze(1)

        target = r + gamma * max_q_prime * done_mask

        target = target.unsqueeze(1)

        loss = F.smooth_l1_loss(q_a, target)
        loss.backward()
        optimizer.step()

def main():
    env = GraspEnv()
    q = Qnet().cuda()
    q_target = Qnet().cuda()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 5
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            input = torch.from_numpy(np.array(s)).float().unsqueeze(dim=0)
            input = input.unsqueeze(dim=0).cuda()
            a = q.sample_action(input, epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 100:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score / print_interval, memory.size(), epsilon * 100))
            
        
        score = 0.0

    env.close()

if __name__ == '__main__':
    main()
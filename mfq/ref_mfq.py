"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
EMBED_SIZE = 50
LR = 0.01                   # learning rate
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
BETA_START = 1                # exploration rate for Boltzmann equation
BETA_END = 0.1
BETA_STEP = (BETA_END-BETA_START)/2000

class Net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim_in, EMBED_SIZE)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(EMBED_SIZE, dim_out)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, n_states=N_STATES, n_actions=N_ACTIONS):
        # self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.eval_net, self.target_net = Net(n_states+1, 1), Net(n_states+1, 1)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES + 4))      # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, _s, _actions, BETA):
        values = []
        for a in _actions:
            x = torch.unsqueeze(torch.tensor(np.append(_s, a), dtype=torch.float), 0)  # add dimension at 0
            values.append(self.eval_net.forward(x).data.numpy().flatten()[0])
        b_val = [val/BETA for val in values]  #Boltzmann, here
        # print(b_val)
        probs = np.exp(b_val) / np.sum(np.exp(b_val))
        # print(probs)
        _a = np.random.choice(len(_actions), p=probs)
        action = _actions[_a]
        return action

    def calculate_mf(self, _s, actions, BETA):
        values = []
        for a in actions:
            x = torch.unsqueeze(torch.tensor(np.append(_s, a), dtype=torch.float), 0)  # add dimension at 0
            values.append(self.target_net.forward(x).data.numpy().flatten()[0])

        b_val = [val / 0.1 for val in values]  # Boltzmann, here
        probs = np.exp(b_val) / np.sum(np.exp(b_val))
        vmf = sum(np.multiply(values, probs))
        return vmf

    def store_transition(self, s, a, r, mf, done):
        transition = np.hstack((s, [a, r, mf], done))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        state_action = torch.tensor(b_memory[:, :N_STATES+1], dtype=torch.float)
        # state = torch.tensor(b_memory[:, :N_STATES], dtype=torch.float)
        # action = torch.tensor(b_memory[:, N_STATES:N_STATES+1],  dtype=torch.long)
        reward = torch.tensor(b_memory[:, N_STATES+1:N_STATES+2], dtype=torch.float)
        detach_state = torch.tensor(b_memory[:, -N_STATES-2:-1], dtype=torch.float)
        mf = torch.tensor(b_memory[:, -2:-1], dtype=torch.float)
        done = torch.tensor(b_memory[:, -1:], dtype=torch.float)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(state_action)  # .gather(1, action)  # shape (batch, 1)
        q_target = reward + GAMMA * mf  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    dqn = DQN()

    print('\nCollecting experience...')
    BETA = BETA_START
    for i in range(1000):
        s = env.reset()

        ep_r = 0
        done = False
        while not done:
            actions = [0, 1]
            a = dqn.choose_action(s, actions, BETA)
            # take action
            s_, r_, done, info = env.step(a)
            if not done:
                mf = dqn.calculate_mf(s_, actions, BETA)
            else:
                r = -20
                mf = 0  # default action
            # modify the reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            dqn.store_transition(s, a, r, mf, done)
            s = s_
            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if BETA > BETA_END:
                    BETA += BETA_STEP

        print('Episode:', i, ' Reward: %i' % int(ep_r))
        print('BETA:', BETA)

    env.close()
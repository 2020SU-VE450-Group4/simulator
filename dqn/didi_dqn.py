import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
from datetime import datetime
from dqn.city_sample import create_city   # here
from simulator.objects import Order


# import os
# print(os.getcwd())

# Hyper Parameters
TOTAL_STEPS = 200000
BATCH_SIZE = 1024
LR = 0.0001                   # learning rate
EPSILON_END = 0.95             # greedy policy
EPSILON_START = 0
EPSILON_STEP = (EPSILON_END-EPSILON_START)/TOTAL_STEPS
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 3000  # target update frequency
MEMORY_CAPACITY = 100000
NUM_GRIDS = 100  # 1322
NUM_TIME_INTERVAL = 48
DIM_STATE = NUM_GRIDS + NUM_TIME_INTERVAL
DIM_ACTION = 2 * NUM_GRIDS
TARGET_DRIVER_ID = 299
directory = os.path.dirname(__file__)


def get_grid_one_hot(idx, n=1322):
    tmp = [0] * n
    tmp[idx] = 1
    return tmp


def get_time_one_hot(t):
    idx = datetime.fromtimestamp(t).hour * 2 + datetime.fromtimestamp(t).minute//30
    tmp = [0] * 48
    tmp[idx] = 1
    return tmp


env = create_city()
grid_map = {id: get_grid_one_hot(i, NUM_GRIDS) for i, id in enumerate(env.grid_ids)}


class Net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Net, self).__init__()
        hidden_layer_unit = [512, 256, 128, 64]
        self.fc1 = nn.Linear(dim_in, hidden_layer_unit[0])
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(hidden_layer_unit[0], hidden_layer_unit[1])
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(hidden_layer_unit[1], hidden_layer_unit[2])
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.fc4 = nn.Linear(hidden_layer_unit[2], hidden_layer_unit[3])
        self.fc4.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(hidden_layer_unit[3], dim_out)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.fc4(x)
        x = nn.functional.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, dim_states, dim_action):
        self.eval_net, self.target_net = Net(dim_states+dim_action, 1), Net(dim_states+dim_action, 1)
        # self.eval_net.load_state_dict(torch.load(directory + "/eval_net"))
        # self.target_net.load_state_dict(torch.load(directory + "/target_net"))

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, dim_states * 2 + dim_action*2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, _s, _actions, epsilon):
        values = []
        for a in _actions:
            x = torch.unsqueeze(torch.tensor(_s + a, dtype=torch.float), 0)  # add dimension at 0
            values.append(self.eval_net.forward(x).data.numpy().flatten()[0])
        if np.random.uniform() < epsilon:  # greedy
            _a = np.argmax(values)
        else:  # random
            _a = random.randint(0, len(_actions)-1)

        return _a

    def choose_action_max(self, _s, actions):
        values = []
        for a in actions:
            x = torch.unsqueeze(torch.tensor(np.append(_s, a), dtype=torch.float), 0)  # add dimension at 0
            values.append(self.eval_net.forward(x))

        # input only one sample
        a_ = np.argmax(values)
        action = actions[a_]
        return action

    def store_transition(self, s, a, r, s_, a_, done):
        transition = np.hstack((s, a, [r], s_, a_, done))
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
        state_action = torch.tensor(b_memory[:, :DIM_STATE+DIM_ACTION], dtype=torch.float)
        reward = torch.tensor(b_memory[:, DIM_STATE+DIM_ACTION:DIM_STATE+DIM_ACTION+1], dtype=torch.float)
        detach_state = torch.tensor(b_memory[:, -DIM_STATE-DIM_ACTION-1:-1], dtype=torch.float)
        done = torch.tensor(b_memory[:, -1], dtype=torch.float)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(state_action)  # .gather(1, action)  # shape (batch, 1)
        q_next = self.target_net(detach_state).detach()     # detach from graph, don't backpropagate
        q_target = reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    dqn = DQN(DIM_STATE, DIM_ACTION)
    end_time = int(time.mktime(datetime.strptime("2016/11/01 13:59:58", "%Y/%m/%d %H:%M:%S").timetuple()))
    # can change the end time here

    print("Start training")

    epsilon = EPSILON_START
    minutes = 30
    for episode in range(50):
        # dicts of current active {drivers: [(loc, time), orders, neighbours]}
        states = env.reset_clean(city_time="2016/11/01 10:00:00")
        start_time = env.city_time

        episode_reward = 0
        busy_drivers = {}
        # driver: [loc, time, action, reward, _loc, _time, _action]
        #   --> ?: may have or may not, used to wait for collecting information to be inserted into the memory

        count = 0
        last_end_node_id = ''
        while env.city_time < end_time:
            # For check use.
            ###########################################################################################################
            # print("Episode is: " + str(episode))
            # print("city time is: " + str(env.city_time))
            # print("Begin to check all drivers in grids.")
            # env.check_all_drivers_in_grids()
            # print("Begin to check all idle drivers in grids.")
            # env.check_idle_drivers_in_grids()
            # print("end one cycle")
            ###########################################################################################################

            count += 1
            if epsilon < EPSILON_END:
                epsilon += EPSILON_STEP

            dispatched_orders = set()   # track orders that are taken
            dispatch_actions = []       # add in dispatch actions to update env
            drivers_to_store = []
            for driver, [(loc, time), orders, drivers] in states.items():
                # For check use.
                #######################################################################################################
                if driver == TARGET_DRIVER_ID:
                    target_driver_start = loc.get_node_index()
                    print("The start node id of the target driver is %s." % target_driver_start)
                    if not last_end_node_id or last_end_node_id == target_driver_start:
                        print("Pass")
                    else:
                        print("Error in driver location.")

                #######################################################################################################
                count += 1
                orders = [o for o in orders if o.order_id not in dispatched_orders]
                idle_order = Order(None, loc, loc, env.city_time, duration=0, price=0)
                orders.append(idle_order)
                neighbours = loc.neighbours[0]
                for nei_grid in neighbours:
                    reposition_duration = 180
                    if loc.get_node_index() in env.transition_trip_time_dict and \
                            nei_grid.get_node_index() in env.transition_trip_time_dict[loc.get_node_index()]:
                        reposition_duration = env.transition_trip_time_dict[loc.get_node_index()][nei_grid.get_node_index()][0]
                    orders.append(Order(None, loc, nei_grid, env.city_time,
                                        reposition_duration, price=0))
                if len(orders):
                    actions = [grid_map[o.get_begin_position_id()] + grid_map[o.get_end_position_id()] for o in orders]
                    aid = dqn.choose_action(grid_map[loc.get_node_index()] + get_time_one_hot(time), actions, epsilon)
                    a = actions[aid]
                    dispatched_orders.add(orders[aid].order_id)

                    # For check use.
                    ###################################################################################################
                    if driver == TARGET_DRIVER_ID:
                        # Get unique node id
                        target_driver_order_end = orders[aid].get_end_position_id()
                        last_end_node_id = target_driver_order_end
                        print("The end node id of the order taken by the target driver is %s." % target_driver_order_end)
                    ###################################################################################################

                    dispatch_actions.append([loc.get_node_index(), driver, orders[aid].get_begin_position_id(),
                                             orders[aid].order_id, orders[aid]])
                    if driver in busy_drivers.keys():  # means it has just finished previous order and become idle again
                        _s = grid_map[loc.get_node_index()] + get_time_one_hot(time)
                        _a = dqn.choose_action_max(_s, actions)
                        # drivers_to_store.append(busy_drivers[driver] + [grid_map[loc.get_node_index()], time])
                        ps, pa, pr = busy_drivers[driver]
                        dqn.store_transition(ps, pa, pr, _s, _a, False)
                    busy_drivers[driver] = [grid_map[loc.get_node_index()] + get_time_one_hot(time), a]
            states, r_, info = env.step(dispatch_actions)
            for driver in r_.keys():
                assert driver in busy_drivers
                busy_drivers[driver].append(r_[driver])
                episode_reward += r_[driver]

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
        print("Episode: ", episode)
        print("Epsilon", epsilon)
        print("Total number of actions inside episode: ", count)
        print("Episode reward", episode_reward)
        print("Response rate", 1 - env.expired_order / env.n_orders)

        # torch.save(dqn.eval_net.state_dict(), directory + "/eval_net")
        # torch.save(dqn.target_net.state_dict(), directory + "/target_net")


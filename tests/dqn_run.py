import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
from datetime import datetime
from dqn.city_sample import create_city   # here
from simulator.objects import Order


TOTAL_STEPS = 400000
BATCH_SIZE = 1024
LR = 0.0001                   # learning rate
EPSILON_END = 0.95             # greedy policy
EPSILON_START = 0
EPSILON_STEP = (EPSILON_END-EPSILON_START)/TOTAL_STEPS
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 3000  # target update frequency
MEMORY_CAPACITY = 80000
NUM_GRIDS = 100  # 1322
NUM_TIME_INTERVAL = 48
DIM_STATE = NUM_GRIDS + NUM_TIME_INTERVAL
DIM_ACTION = 2 * NUM_GRIDS
TARGET_DRIVER_ID = 299
global memory_counter
memory_counter = 0
memory = np.zeros((MEMORY_CAPACITY, DIM_STATE * 2 + DIM_ACTION*2 + 2))
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


# def test(net):
#     if memory_counter > MEMORY_CAPACITY:
#         sample_index = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
#     else:
#         sample_index = np.random.choice(memory_counter, size=BATCH_SIZE)
#     # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
#     b_memory = memory[sample_index, :]
#     state_action = torch.tensor(b_memory[:, :DIM_STATE + DIM_ACTION], dtype=torch.float)
#     reward = torch.tensor(b_memory[:, DIM_STATE + DIM_ACTION:DIM_STATE + DIM_ACTION + 1], dtype=torch.float)
#     detach_state = torch.tensor(b_memory[:, -DIM_STATE - DIM_ACTION - 1:-1], dtype=torch.float)
#     done = torch.tensor(b_memory[:, -1], dtype=torch.float)
#
#     # q_eval w.r.t the action in experience
#     q_eval = net(state_action)  # .gather(1, action)  # shape (batch, 1)


def choose_action(_s, _actions, net):
    values = []
    for a in _actions:
        x = torch.unsqueeze(torch.tensor(_s + a, dtype=torch.float), 0)  # add dimension at 0
        values.append(net.forward(x).data.numpy().flatten()[0])
    _a = np.argmax(values)

    return _a


def choose_action_max(_s, actions, net):
    values = []
    for a in actions:
        x = torch.unsqueeze(torch.tensor(np.append(_s, a), dtype=torch.float), 0)  # add dimension at 0
        values.append(net.forward(x))

    # input only one sample
    a_ = np.argmax(values)
    action = actions[a_]
    return action


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


if __name__ == '__main__':
    net = Net(DIM_STATE+DIM_ACTION, 1)
    net.load_state_dict(torch.load('./Archive/eval_net'))
    end_time = int(time.mktime(datetime.strptime("2016/11/01 13:30:58", "%Y/%m/%d %H:%M:%S").timetuple()))
    # can change the end time here

    print("Start testing")

    minutes = 30
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
        dispatched_orders = set()   # track orders that are taken
        dispatch_actions = []       # add in dispatch actions to update env
        drivers_to_store = []
        for driver, [(loc, time), orders, drivers] in states.items():
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
                aid = choose_action(grid_map[loc.get_node_index()] + get_time_one_hot(time), actions, net)
                a = actions[aid]
                dispatched_orders.add(orders[aid].order_id)

                dispatch_actions.append([loc.get_node_index(), driver, orders[aid].get_begin_position_id(),
                                             orders[aid].order_id, orders[aid]])
                if driver in busy_drivers.keys():  # means it has just finished previous order and become idle again
                    _s = grid_map[loc.get_node_index()] + get_time_one_hot(time)
                    _a = choose_action_max(_s, actions, net)
                    # drivers_to_store.append(busy_drivers[driver] + [grid_map[loc.get_node_index()], time])
                    ps, pa, pr = busy_drivers[driver]
                    # store_transition(ps, pa, pr, _s, _a, False)
                    count += 1
                busy_drivers[driver] = [grid_map[loc.get_node_index()] + get_time_one_hot(time), a]
        states, r_, info = env.step(dispatch_actions)
        for driver in r_.keys():
            assert driver in busy_drivers
            busy_drivers[driver].append(r_[driver])
            episode_reward += r_[driver]
        print(r_)



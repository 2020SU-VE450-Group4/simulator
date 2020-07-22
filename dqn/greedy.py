import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
from datetime import datetime
from dqn.city_sample import create_city   # here
from simulator.objects import Order

TOTAL_STEPS = 80000
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

if __name__ == '__main__':
    end_time = int(time.mktime(datetime.strptime("2016/11/01 13:59:58", "%Y/%m/%d %H:%M:%S").timetuple()))
    print("Start training")

    epsilon = EPSILON_START
    minutes = 30
    for episode in range(30):
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

            dispatched_orders = set()  # track orders that are taken
            dispatch_actions = []  # add in dispatch actions to update env
            drivers_to_store = []
            for driver, [(loc, time), orders, drivers] in states.items():
                # For check use.
                #######################################################################################################
                # if driver == TARGET_DRIVER_ID:
                #     target_driver_start = loc.get_node_index()
                #     print("The start node id of the target driver is %s." % target_driver_start)
                #     if not last_end_node_id or last_end_node_id == target_driver_start:
                #         print("Pass")
                #     else:
                #         print("Error in driver location.")

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
                        reposition_duration = \
                        env.transition_trip_time_dict[loc.get_node_index()][nei_grid.get_node_index()][0]
                    orders.append(Order(None, loc, nei_grid, env.city_time,
                                        reposition_duration, price=0))
                if len(orders):
                    actions = [grid_map[o.get_begin_position_id()] + grid_map[o.get_end_position_id()] for o in orders]
                    # aid = dqn.choose_action(grid_map[loc.get_node_index()] + get_time_one_hot(time), actions, epsilon)
                    max_reward = 0
                    aid = 0
                    for i, order in enumerate(orders):
                        # print(order.get_price())
                        max_reward = max(max_reward, order.get_price())
                        if max_reward == order.get_price():
                            aid = i
                            # print(aid)
                    dispatched_orders.add(orders[aid].order_id)
                    a = actions[aid]

                    # For check use.
                    ###################################################################################################
                    # if driver == TARGET_DRIVER_ID:
                    #     # Get unique node id
                    #     target_driver_order_end = orders[aid].get_end_position_id()
                    #     last_end_node_id = target_driver_order_end
                    #     print(
                    #         "The end node id of the order taken by the target driver is %s." % target_driver_order_end)
                    ###################################################################################################

                    dispatch_actions.append([loc.get_node_index(), driver, orders[aid].get_begin_position_id(),
                                             orders[aid].order_id, orders[aid]])

                    busy_drivers[driver] = [grid_map[loc.get_node_index()] + get_time_one_hot(time), a]
            states, r_, info = env.step(dispatch_actions)
            for driver in r_.keys():
                assert driver in busy_drivers
                busy_drivers[driver].append(r_[driver])
                episode_reward += r_[driver]


            # if dqn.memory_counter > MEMORY_CAPACITY:
            #     dqn.learn()
        print("Episode: ", episode)
        print("Epsilon", epsilon)
        print("Total number of actions inside episode: ", count)
        print("Episode reward", episode_reward)
        print("Response rate", 1 - env.expired_order / env.n_orders)

        # torch.save(dqn.eval_net.state_dict(), directory + "/eval_net")
        # torch.save(dqn.target_net.state_dict(), directory + "/target_net")


from simulator.envs import *
import json
import pickle
# to-do list in priority order
# TODO: define a pick-up dictionary
# TODO: Real order injection
# TODO: specify idle driver transition
# TODO: coodinate based version
# TODO: Add new online idle drivers within the day
# TODO: Specify more state/reward/action interface

def random_dispatch(s):
    """
    State input given by get_observation_verbose
    """
    action = []  # action as the form driver's grid id, driver id, order's grid id, order id, order (for virtual)
    for grid_id, state in s.items():
        orders, drivers = state
        pair_length = min(len(orders), len(drivers))
        for i in range(pair_length):
            driver = drivers[i]
            if driver.get_driver_id() == 641:
                driver.print_driver()
            order = orders[i]
            if driver.get_driver_id() == 641:
                order.print_order()
            action.append(
                [driver.node.get_node_index(), driver.get_driver_id(), order.get_begin_position().get_node_index(),
                 order.order_id, None])
    return action

# load all needed files
with open("all_grid.pkl", "rb") as pk:
    all_grids = pickle.load(pk)

with open("grid_neighbour.json", "r") as fp:
    neighbour_dict = json.load(fp)

with open("20161101_demand_dict_grid_10min", "rb") as pk:
    order_num_dist = pickle.load(pk)

with open("trans_prob.json", "r") as fp:
    transition_prob_dict = json.load(fp)

with open("duration.json", "r") as fp:
    transition_trip_time_dict = json.load(fp)

with open("reward.json", "r") as fp:
    transition_reward_dict = json.load(fp)

with open("driver_distribution_dict.pkl", "rb") as pk:
    init_idle_driver = pickle.load(pk)

with open("time_distribution_1000.pkl", "rb") as pk:
    time_dist = pickle.load(pk)

end_time = int(time.mktime(datetime.strptime("2016/11/01 11:29:58", "%Y/%m/%d %H:%M:%S").timetuple()))  # can change the end time here
myCity = CityReal(all_grids, neighbour_dict, "2016/11/01 10:00:00", False, False, order_num_dist, transition_prob_dict, transition_trip_time_dict, transition_reward_dict,
                 init_idle_driver, time_dist)

for episode in range(1):
    s = myCity.reset_clean(city_time="2016/11/01 10:00:00")
    episode_reward = 0
    while True:
        # write a simple pairing within each grid here
        action = []
        s_, reward, info = myCity.step(action)
        print(reward)
        s = s_
        episode_reward += reward
        print(myCity.city_time)
        if myCity.city_time >= end_time:
            break
    print("Episode reward", episode_reward)
    print("Response rate", myCity.expired_order/myCity.n_orders)

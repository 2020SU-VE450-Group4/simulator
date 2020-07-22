from simulator.envs import *
import json
import pickle


def create_city():
    directory = os.path.dirname(__file__)
    directory = os.path.dirname(directory) + '/tests'
    # load all needed files
    with open(directory + "/all_grid.pkl", "rb") as pk:
        all_grids = pickle.load(pk)

    with open(directory + "/grid_neighbour.json", "r") as fp:
        neighbour_dict = json.load(fp)

    with open(directory + "/20161101_demand_dict_grid_10min", "rb") as pk:
        order_num_dist = pickle.load(pk)

    with open(directory + "/trans_prob.json", "r") as fp:
        transition_prob_dict = json.load(fp)

    with open(directory + "/duration.json", "r") as fp:
        transition_trip_time_dict = json.load(fp)

    with open(directory + "/reward.json", "r") as fp:
        transition_reward_dict = json.load(fp)

    with open(directory + "/driver_distribution_dict.pkl", "rb") as pk:
        init_idle_driver = pickle.load(pk)

    with open(directory + "/time_distribution_1000.pkl", "rb") as pk:
        time_dist = pickle.load(pk)
        time_dist = time_dist[0:40]  # decrease the overall working time of the drivers, since we have too many drivers
        s = sum(time_dist)
        for i in range(len(time_dist)):
            time_dist[i] /= s

    with open(directory + "/real_order_20161101.pkl", "rb") as pk:
        real_order_list = pickle.load(pk)

    return CityReal(all_grids, neighbour_dict, "2016/11/01 10:00:00", real_bool=True, coordinate_based=False,
                    order_num_dist=order_num_dist, transition_prob_dict=transition_prob_dict,
                    transition_trip_time_dict=transition_trip_time_dict, transition_reward_dict=transition_reward_dict,
                    init_idle_driver=init_idle_driver, working_time_dist=time_dist, real_orders=real_order_list,
                    order_generation_interval=600, driver_online_interval=600)


import os, time
from datetime import datetime
from random import sample 

os.getcwd()   
from simulator.envs import CityReal
from model.km import  *
from simulator import envs
from model import km
import json
import pickle
import argparse


def get_dispatch_observ(s, num_sample):
    
    dispatch_observ = []
    order_dict = {}
    order_dict_pos = {}
    driver_dict = {}
    for grid_id, state in s.items():
        _, orders, drivers = state
        dispatch_observ = order_driver_bigraph(orders, drivers, dispatch_observ, num_sample)
        for d in drivers:
            driver_dict[d._driver_id] = d.node.get_node_index()
        for o in orders:
            order_dict_pos[o.order_id] = o.get_begin_position_id()
            order_dict[o.order_id] = o
     
    return dispatch_observ, driver_dict, order_dict_pos, order_dict


def dispatch(s, num_sample=0.3):
    """
    State input given by get_observation_verbose
    """
    dispatch_observ, driver_dict, order_dict_pos, order_dict = get_dispatch_observ(s, num_sample)
    res = km_dispatch(dispatch_observ)
    
    dispatch_action = []
    for order_driver_v in res:
        dispatch_action.append([driver_dict[order_driver_v[1]], order_driver_v[1], order_dict_pos[order_driver_v[0]], order_driver_v[0], order_dict[order_driver_v[0]]  ])
    
    # action as the form driver's grid id, driver id, order's grid id, order id, order (for virtual)
    return dispatch_action


def order_driver_bigraph(orders, drivers, dispatch_observ, num_sample):
    for o in orders:
        # TODO: sample more drivers
        sample_num = int( num_sample * len(drivers))
        for d in sample(drivers, sample_num):
            pair = {}
            pair['order_id'] = o.order_id
            pair['driver_id'] = d._driver_id
            pair['order_start_grid'] = o.get_begin_position_id()
            pair['order_finish_grid'] = o._end_p.get_node_index()
            pair['driver_grid'] = d.node.get_node_index()
            pair['duration'] = o.get_duration()
            pair['price'] = o.get_price()
            pair['begin_time'] = o.get_begin_time()
            pair['cur_t'] = calc_tp(pair['begin_time'])
            pair['dst_t'] = calc_tp(pair['begin_time'] + int(pair['duration'] ))
            pair['waiting_t'] = o.get_wait_time()
            pair['cancel_rate'] = 0 # wait to be calculate
            # lack of coordinate info
            pair['order_driver_distance'] = 0 if pair['driver_grid'] == pair['order_start_grid'] else 0
            pair['pick_up_eta'] = pair['order_driver_distance']
            pair['order'] = o
            dispatch_observ.append(pair)
    return dispatch_observ

def cal_reward(observ):
    dur = int(observ['duration'])
    rw_unit = observ['price'] / dur
    gamma = 0.9
    rw = sum([rw_unit *  gamma ** i for i in range(dur)])
    
    h_cur, h_dst = observ['order_start_grid'], observ['order_finish_grid']
    t_cur, t_dst = observ['cur_t']  , observ['dst_t']
    
    adv = value_map[h_dst][t_dst] * gamma**dur + rw - value_map[h_cur][t_cur]
    # cancel_rate = observ['cancel_rate'] 
    return adv


def km_dispatch( dispatch_observ):
      order_driver_rw = [(x['order_id'], x['driver_id'],  cal_reward(x)) for x in dispatch_observ]
      process = KuhnMunkres()
      process.set_matrix(order_driver_rw)
      process.km()
      res = process.get_connect_result()
      return res

 
def calc_tp(timestamp):
    tp = 300
    h = datetime.fromtimestamp(timestamp).hour
    m = datetime.fromtimestamp(timestamp).minute
    s = datetime.fromtimestamp(timestamp).second

    return int((s + m * 60 + h * 60 * 60) / tp)

def print_state(s): 
    for grid_id, state in s.items():
        # [(loc, time), orders, neighbour_drivers]
        (loc, time), orders, drivers = state
        print("Orders/ Drivers from Grid ", grid_id)
        # ("Driver:", self._driver_id, self.node.get_node_index(), self.online, self.onservice, self.offline_time)
        for o in orders:
            o.print_order()
        # ("Order:", self.order_id, self._begin_p.get_node_index(), self._end_p.get_node_index(), self._begin_t, self._t, self._p)
        for d in drivers:
            d.print_driver()
        print("Number of drivers", len(drivers))
        print("Number of orders", len(orders))

def main():
    os.chdir('tests/')
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
    
    # with open("driver_distribution_dict.pkl", "rb") as pk:
    with open("20161101_drivers", "rb") as pk:
        init_idle_driver = pickle.load(pk)
    
    with open("time_distribution_1000.pkl", "rb") as pk:
        time_dist = pickle.load(pk)
    
    with open("real_order_20161101.pkl", "rb") as pk:
        real_order_list = pickle.load(pk)

    parser = argparse.ArgumentParser(description='Planning and Learning dispatch')
    parser.add_argument('--value', type=str, default="V0104mean.pkl",
                        help='state value *.pkl') 
    parser.add_argument('--sample', type=int, default=0.3,
                        help='number of sample drivers for km') 
    parser.add_argument('--local', type=bool, default=False,
                        help='local test ot real world') 
    args = parser.parse_args()
    with open(args.value, "rb") as pk:
        global value_map
        value_map = pickle.load(pk)
    
    args = parser.parse_args()
    os.chdir('../')
    
    end_time = int(time.mktime(datetime.strptime("2016/11/01 11:29:58", "%Y/%m/%d %H:%M:%S").timetuple()))  # can change the end time here
    myCity = CityReal(all_grids, neighbour_dict, "2016/11/01 10:00:00", real_bool=args.local, coordinate_based=False, order_num_dist=order_num_dist,
                      transition_prob_dict=transition_prob_dict, transition_trip_time_dict=transition_trip_time_dict, transition_reward_dict=transition_reward_dict,
                     init_idle_driver=init_idle_driver, working_time_dist=time_dist, real_orders=real_order_list)
    
    for episode in range(1):
        s = myCity.reset_clean(city_time="2016/11/01 10:00:00")
        
        episode_reward = 0
        while True:
            print("Time: ", myCity.city_time )
            print("Time: ", myCity.city_time, file=fnew_out)
            print_state(s) 
            # write a simple pairing within each grid here
            action = dispatch(s, args.sample)
            print("Action: ", action)
            print("Action: ", action, file=fnew_out)
            s_, reward, info = myCity.step(action)
            print(reward)
            print(reward, file=fnew_out)
            s = s_ 
            
            if isinstance(reward, dict):
                global_reward = myCity.get_global_reward(reward)
                episode_reward += global_reward
            else:
                episode_reward += reward
            if myCity.city_time >= end_time:
                break
        print("Episode reward", episode_reward)
        print("Episode reward", episode_reward, file=fnew_out)
        print("Response rate", myCity.expired_order/myCity.n_orders)
        print("Response rate", myCity.expired_order / myCity.n_orders, file=fnew_out)


if __name__ == '__main__':
    fnew_out = open("new_output.txt", 'w+')
    main()
    fnew_out.close()
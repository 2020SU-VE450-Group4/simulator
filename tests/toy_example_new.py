from dqn.city import create_city
import time
from datetime import datetime

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


end_time = int(time.mktime(datetime.strptime("2016/11/01 11:29:58", "%Y/%m/%d %H:%M:%S").timetuple()))  # can change the end time here
myCity = create_city()

for episode in range(1):
    s = myCity.reset_clean(city_time="2016/11/01 10:00:00")
    episode_reward = 0
    while True:
        # write a simple pairing within each grid here
        action = []
        s_, reward, info = myCity.step(action)
        print(reward)
        s = s_
        if isinstance(reward, dict):
            global_reward = myCity.get_global_reward(reward)
            episode_reward += global_reward
        else:
            episode_reward += reward
        print(myCity.city_time)
        if myCity.city_time >= end_time:
            break
    print("Episode reward", episode_reward)
    print("Response rate", myCity.expired_order / myCity.n_orders)

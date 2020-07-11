import os, sys, random, time, datetime
import logging
sys.path.append("../")

from simulator.objects import *
from simulator.utilities import *
# from algorithm import *

# current_time = time.strftime("%Y%m%d_%H-%M")
# log_dir = "/nfs/private/linkaixiang_i/data/dispatch_simulator/experiments/"+current_time + "/"
# mkdir_p(log_dir)
# logging.basicConfig(filename=log_dir +'logger_env.log', level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger_ch = logging.StreamHandler()
logger_ch.setLevel(logging.DEBUG)
logger_ch.setFormatter(logging.Formatter(
    '%(asctime)s[%(levelname)s][%(lineno)s:%(funcName)s]||%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(logger_ch)
RANDOM_SEED = 0  # unit test use this random seed.


class CityReal:

    def __init__(self, all_grids, neighbour_dict, start_time_string, real_bool, coordinate_based, order_num_dist, transition_prob_dict, transition_trip_time_dict, transition_reward_dict,
                 init_idle_driver, working_time_dist, probability=1.0/28, real_orders="", order_generation_interval=600):
        """
        :param all_grids: a list of hexagon grid ids
        :param neighbour_dict: a dict of {gridID: neighbour grid list} of radius 1 and 2
        :param start_time_string: a string of the start time of the simulator
        :param real_bool: a boolean value represent whether the order is real (True, the same as the historical data) or bootstrapped (False, can vary greatly)
        :param coordinate_based: a boolean value represent whether the simulator is a grid based one or coordinate base one
        :param mapped_matrix_int: 2D matrix: each position is either -100 or grid id from order in real data.
        :param order_num_dist: {time:{gridID: count, ...}}
        :param transition_prob_dict: a dict of transition probability for user orders between grids
        :param transition_trip_time_dict: a dict of transition time for user orders between grids
        :param transition_reward_dict: a dict of reward value for user orders between grids
        :param init_idle_driver: a dict of {nodeID: idle_driver_num, ...}
        :param working_time_dist: distribution of working time of drivers
        :param idle_driver_dist_time: [[mu1, std1], [mu2, std2], ..., [mu144, std144]] mean and variance of idle drivers in
        the city at each time
        :param idle_driver_location_mat: 144 x num_valid_grids matrix.
        :param order_time_dist: [ 0.27380797,..., 0.00205766] The probs of order duration = 1 to 9
        :param order_price_dist: [[10.17, 3.34],   # mean and std of order's price, order durations = 10 minutes.
                                   [15.02, 6.90],  # mean and std of order's price, order durations = 20 minutes.
                                   ...,]
        :param onoff_driver_location_mat: 144 x 504 x 2: 144 total time steps, num_valid_grids = 504.
        mean and std of online driver number - offline driver number
        onoff_driver_location_mat[t] = [[-0.625       2.92350389]  <-- Corresponds to the grid in target_node_ids
                                        [ 0.09090909  1.46398452]
                                        [ 0.09090909  2.36596622]
                                        [-1.2         2.05588586]...]
        :param M:
        :param N:
        :param n_side:
        :param time_interval:
        :param l_max: The max-duration of an order
        :return:
        """
        self.coordinate_based = coordinate_based
        self.grid_ids = list(all_grids)
        self.grids = {node_id: Node(node_id) for node_id in self.grid_ids}  # a dict of grid objects
        self.n_grids = len(self.grid_ids)

        for grid_id, grid in self.grids.items():
            neighbour = []
            for radius in range(2):
                neighbour.append([])
                for idx in range(len(neighbour_dict[grid_id][radius])):
                    neighbour[radius].append(self.grids[neighbour_dict[grid_id][radius][idx]])
            grid.set_neighbours(neighbour)

        self.start_time = start_time_string  # e.g. "2016/11/01 10:00:00"
        self.city_time = int(time.mktime(datetime.strptime(self.start_time, "%Y/%m/%d %H:%M:%S").timetuple()))
        self.order_response_rate = 0

        self.RANDOM_SEED = RANDOM_SEED

        self.n_orders = 0  # total number of existed orders, id start from 0, used to calculate the order_id
        self.order_num_dist = order_num_dist
        self.real_bool = real_bool
        self.order_generation_interval = order_generation_interval
        self.transition_prob_dict = transition_prob_dict
        self.transition_trip_time_dict = transition_trip_time_dict
        self.transition_reward_dict = transition_reward_dict
        self.distribution_name = "Poisson"
        self.expired_order = 0

        self.drivers = {}  # driver[driver_id] = driver_instance, driver_id start from 0
        self.onservice_drivers = {}
        self.n_drivers = 0  # total number of drivers
        self.n_offline_drivers = 0  # total number of offline drivers.
        self.init_idle_driver = init_idle_driver
        self.working_time_dist = working_time_dist

        self.real_orders = real_orders


        self.p = probability   # sample probability

        self.day_orders = []  # one day's order.

    def get_observation(self):
        next_state = np.zeros((2, self.n_grids))   # 原来的代码像CNN一样活着，我们就暂时不必了，我们直接铺开。。。不用geographical info了
        for idx, grid_id in enumerate(self.grid_ids):
            # if grid is not None:
            # row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
            next_state[0, idx] = self.grids[grid_id].idle_driver_num
            next_state[1, idx] = self.grids[grid_id].order_num
        return next_state

    def get_observation_verbose(self):
        """ Return all active orders and drivers for all nodes
        """
        state = {}
        for grid_id, grid in self.grids.items():
            o = grid.get_active_orders(self.city_time)
            d = list(grid.get_idle_drivers().values())
            state[grid_id] = [o,d]
        return state

    def get_observation_neighbour(self):
        """ Return the observation containing each idle driver's location, time and the active orders and idle drivers inside its neighbour
        """
        state = {}
        for grid_id, grid in self.grids.items():
            drivers = list(grid.get_idle_drivers().values())
            orders = self.get_active_order_neighbour(grid)
            neighbour_drivers = self.get_active_driver_neighbour(grid)
            for driver in drivers:
                driver_id = driver.get_driver_id()
                loc = driver.get_position()
                time = driver.city_time
                neighbour_drivers.remove(driver)  # pop itself
                assert driver_id not in state
                state[driver_id] = [(loc, time), orders, neighbour_drivers]
                neighbour_drivers.append(driver)  # insert back
        return state

    def get_active_order_neighbour(self, grid):
        neighbour = [grid] + grid.neighbours[0]
        orders = []
        for g in neighbour:
            orders += g.get_active_orders(self.city_time)
        return orders

    def get_active_driver_neighbour(self, grid):
        neighbour = [grid] + grid.neighbours[0] + grid.neighbours[1]
        drivers = []
        for g in neighbour:
            drivers += list(g.get_idle_drivers().values())
        return drivers

    def get_num_idle_drivers(self):
        """ Compute idle drivers
        :return:
        """
        city_idle_drivers_count = 0
        for grid in list(self.grids.values()):
            if grid is not None:
                city_idle_drivers_count += grid.idle_driver_num
        return city_idle_drivers_count

    def get_observation_driver_state(self):
        """ Get idle driver distribution, computing #drivers from node.
        :return:
        """
        next_state = np.zeros(self.n_grids)
        grids = list(self.grids.values())
        for idx, grid in enumerate(grids):
            if grid is not None:
                next_state[idx] = grid.get_idle_driver_numbers_loop()
        return next_state

    def get_reward_at_setoff(self, dispatched_drivers):
        reward = 0
        for driver in dispatched_drivers:
            ontrip_order = driver.get_order()
            reward += ontrip_order.get_price()
        return reward

    def get_reward_average(self):
        reward = 0
        for driver_id, driver in self.onservice_drivers.items():
            ontrip_order = driver.get_order()
            assert ontrip_order is not None
            reward += ontrip_order.get_price() / ontrip_order.get_duration()
        return reward

    def get_reward_per_driver(self, dispatched_drivers):
        reward_dict = {}
        for driver in dispatched_drivers:
            reward_dict[driver.get_driver_id()] = driver.get_order().get_price()
        return reward_dict

    @staticmethod
    def get_global_reward(reward_dict):
        """ Record the accumulative reward city-wide
        """
        return sum(reward_dict.values())

    def reset_randomseed(self, random_seed):
        self.RANDOM_SEED = random_seed

    def reset(self):
        """ Return initial observation: get order distribution and idle driver distribution

        """
        # initialization drivers according to the distribution at time 0
        self.utility_add_driver_real_new()

        # generate orders at first time step
        if self.real_bool is False:
            # Init orders of current time step
            moment = int(self.city_time / self.order_generation_interval)
            self.step_bootstrap_order(self.order_num_dist[moment])
        else:
            self.utility_real_oneday_order()
        return self.get_observation()

    def reset_clean(self, generate_order_real=False, city_time=""):
        """ 1. Set city time
            2. clean current drivers and orders, regenerate new orders and drivers.
            can reset anytime, usually at the start of each episode

        :param: generate_order_real bool generate order based on num_dict or
        :return:
        """
        if city_time != "":
            self.city_time = int(time.mktime(datetime.strptime(self.start_time, "%Y/%m/%d %H:%M:%S").timetuple()))

        # clean orders and drivers
        self.n_orders = 0
        self.expired_order = 0
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
        self.n_drivers = 0  # total idle number of drivers. online and not on service.
        self.n_offline_drivers = 0  # total number of offline drivers.
        for key, grid in self.grids.items():
            if grid is not None:
                grid.clean_node()

        # Generate order.
        if generate_order_real is False:
            # Init orders of current time step
            moment = int(self.city_time / self.order_generation_interval)
            self.step_bootstrap_order(self.order_num_dist[moment])
        else:
            self.utility_real_oneday_order()
        # Init current driver distribution
        self.utility_add_driver_real_new()
        return self.get_observation_neighbour()

    def step_add_finished_drivers(self):
        """ Deal with finished orders, check driver status. finish order, add then to the destination node
            :return:
        """
        keys = list(self.onservice_drivers.keys())
        for driver_id in keys:
            driver = self.onservice_drivers[driver_id]
            assert driver.city_time == self.city_time
            assert driver.onservice is True
            assert driver.online is True
            order_end_time = driver.order.get_assigned_time() + driver.order.get_duration()
            if driver.city_time >= order_end_time:
                driver.set_position(driver.order.get_end_position())
                driver.set_order_finish()
                driver.node.add_driver(driver.get_driver_id(), driver)
                self.onservice_drivers.pop(driver_id)
            elif self.city_time < order_end_time:
                pass

    def step_driver_offline_nodewise(self):
        """ node wise control driver online offline
        :return:
        """
        for grid_id, grid in self.grids.items():
            if grid is not None:
                grid.set_expired_driver_offline(self.city_time)

    def step_bootstrap_order(self, order_num_dict):
        new_order_count = 0
        for grid_id in order_num_dict.keys():
            start_grid_id = grid_id
            start_grid = self.grids[start_grid_id]
            demand_count_vec = np.random.poisson(order_num_dict[grid_id] / self.order_generation_interval, self.order_generation_interval)
            for i in range(len(demand_count_vec)):
                if demand_count_vec[i] > 0:
                    destination_targets = list(self.transition_prob_dict[start_grid_id].keys())
                    p = list(self.transition_prob_dict[start_grid_id].values())
                    end_grid_ids = np.random.choice(destination_targets, demand_count_vec[i], p=p)
                    for end_grid_id in end_grid_ids:
                        end_grid = self.grids[end_grid_id]
                        trip_time = self.transition_trip_time_dict[start_grid_id][end_grid_id][0]  # temporarily use mean to stablize it  # TODO: change to normal
                        price = self.transition_reward_dict[start_grid_id][end_grid_id][0]   # also use mean
                        start_grid.add_order(new_order_count + self.n_orders, self.city_time + i, end_grid, trip_time, price)  # can specify wait time here
                        new_order_count += 1
        self.n_orders += new_order_count

    def step_remove_unfinished_orders(self):
        for grid_id, grid in self.grids.items():
            if grid is not None:
                self.expired_order += grid.remove_unfinished_order(self.city_time)

    def step_pre_order_assigin(self, next_state):

        remain_drivers = next_state[0] - next_state[1]
        remain_drivers[remain_drivers < 0] = 0

        remain_orders = next_state[1] - next_state[0]
        remain_orders[remain_orders < 0] = 0

        if np.sum(remain_orders) == 0 or np.sum(remain_drivers) == 0:
            context = np.array([remain_drivers, remain_orders])
            return context

        remain_orders_1d = remain_orders.flatten()
        remain_drivers_1d = remain_drivers.flatten()

        for node in self.nodes:
            if node is not None:
                curr_node_id = node.get_node_index()
                if remain_orders_1d[curr_node_id] != 0:
                    for neighbor_node in node.neighbors:
                        if neighbor_node is not None:
                            neighbor_id = neighbor_node.get_node_index()
                            a = remain_orders_1d[curr_node_id]
                            b = remain_drivers_1d[neighbor_id]
                            remain_orders_1d[curr_node_id] = max(a-b, 0)
                            remain_drivers_1d[neighbor_id] = max(b-a, 0)
                        if remain_orders_1d[curr_node_id] == 0:
                            break

        context = np.array([remain_drivers_1d.reshape(self.M, self.N),
                   remain_orders_1d.reshape(self.M, self.N)])
        return context

    def step_dispatch(self, dispatch_actions):
        """ Execute dispatch actions
        :param dispatch_actions: in the form of (driver_grid_id, driver_id, order_grid_id, order_id, order)
                                 if the order is an idle action or a reposition action,
                                    order_id is None, and order object is used to specify the destination (an self-created order object!)
        """
        dispatched_drivers = []
        for action in dispatch_actions:
            driver_grid_id, driver_id, order_grid_id, order_id, order = action
            if driver_grid_id not in self.grids:
                print(driver_grid_id)
                raise ValueError('Step_dispatch: Driver grid id error')
            driver_grid = self.grids[driver_grid_id]
            if driver_id not in driver_grid.drivers:
                raise ValueError('Step_dispatch: Dispatched a driver not in the grid')
            driver = driver_grid.drivers[driver_id]

            if order_grid_id not in self.grids:
                raise ValueError('Step_dispatch: Order grid id error')
            order_grid = self.grids[order_grid_id]
            if order_id is not None and order_id not in order_grid.orders:  # if order_id is None, it is idle or reposition action
                raise ValueError('Step_dispatch: Assigned order does not exist in the grid')
            elif order_id is not None:
                order_ = order_grid.orders[order_id]
                assert order_ == order

            # deal with pick-up process
            if driver_grid_id != order_grid_id:
                original_duration = order.get_duration()
                pickup_duration = 180
                if driver_grid_id in self.transition_trip_time_dict and \
                        order_grid_id in self.transition_trip_time_dict[driver_grid_id]:
                    pickup_duration = self.transition_trip_time_dict[driver_grid_id][order_grid_id][0]
                order.set_duration(original_duration + pickup_duration)
            # deal with order cancellation
            if order_id is not None and np.random.uniform() < 0.05783:  # the passenger cancel the order
                order.set_duration(0)
                order.set_begin_position(driver_grid)
                order.set_end_position(driver_grid)
                order.set_price(0)
            driver.take_order(order)  # take order normally
            dispatched_drivers.append(driver)
            order.set_assigned_time(self.city_time)
            self.onservice_drivers[driver_id] = driver
            driver_grid.remove_driver(driver_id)
            if order_id is not None:  # the assigned order is a real order
                order_grid.remove_dispatched_order(order_id)

        reward = self.get_reward_per_driver(dispatched_drivers)
        return reward

    def step_increase_city_time(self):
        self.city_time += 60
        # set city time of drivers
        for driver_id, driver in self.drivers.items():
            driver.set_city_time(self.city_time)

    def step(self, dispatch_actions):
        info = []
        # Loop over all dispatch action, change the drivers and nodes accordingly
        """    T      """
        reward = self.step_dispatch(dispatch_actions)

        # increase city time
        self.step_increase_city_time()
        """    T  +   2s     """
        self.step_add_finished_drivers()  # drivers finish order become available again and added to new grid
        # Generate new orders
        if self.real_bool is False and self.city_time % self.order_generation_interval == 0:
            # Init orders of current time step
            moment = int(self.city_time / self.order_generation_interval)
            if moment not in self.order_num_dist:
                raise KeyError("KeyError: the current moment does not exist in order_num_dist")
            self.step_bootstrap_order(self.order_num_dist[moment])
        # TODO: add new drivers

        self.step_driver_offline_nodewise()
        self.step_remove_unfinished_orders()   # remove the orders with
        next_state = self.get_observation_neighbour()
        return next_state, reward, info

    def utility_add_driver_real_new(self):
        n_total_drivers = len(self.drivers.keys())
        new_driver_count = 0
        for grid_id, value in self.init_idle_driver.items():
            for i in range(value):
                added_driver_id = n_total_drivers + new_driver_count
                new_driver_count += 1
                online_duration = np.random.choice(range(1, len(self.working_time_dist)+1), p=self.working_time_dist) * 1000
                self.drivers[added_driver_id] = Driver(added_driver_id, self.city_time, self.city_time+online_duration)
                self.drivers[added_driver_id].set_position(self.grids[grid_id])
                self.grids[grid_id].add_driver(added_driver_id, self.drivers[added_driver_id])
        self.n_drivers += new_driver_count

    def utility_real_oneday_order(self):
        new_order_count = 0
        for order in self.real_orders:
            # here order is in the format of start_grid_id, start_time, end_grid_id, trip_time, price, start_lng, start_lat, end_lng, end_lat
            if int(order[1]) >= self.city_time:
                start_grid = self.grids[order[0]]
                end_grid = self.grids[order[2]]
                start_grid.add_order(new_order_count + self.n_orders, int(order[1]), end_grid, int(order[3]), float(order[4]))  # can specify wait time here
                new_order_count += 1



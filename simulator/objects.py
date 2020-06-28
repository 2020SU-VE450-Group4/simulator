import numpy as np
from abc import ABCMeta, abstractmethod

class Distribution(metaclass=ABCMeta):
    ''' Define the distribution from which sample the orders'''
    @abstractmethod
    def sample(self):
        pass

class PoissonDistribution(Distribution):

    def __init__(self, lam):
        self._lambda = lam

    def sample(self, seed=0):
        np.random.seed(seed)
        return np.random.poisson(self._lambda, 1)[0]


class GaussianDistribution(Distribution):

    def __init__(self, args):
        mu, sigma = args
        self.mu = mu        # mean
        self.sigma = sigma  # standard deviation

    def sample(self, seed=0):
        np.random.seed(seed)
        return np.random.normal(self.mu, self.sigma, 1)[0]


class Node(object):
    __slots__ = ('neighbours', '_index', 'orders', 'drivers',
                 'order_num', 'idle_driver_num', 'offline_driver_num'
                 'order_generator', 'offline_driver_num', 'order_generator',
                 'n_side', 'layers_neighbors', 'layers_neighbors_id')

    def __init__(self, index):
        # private
        self._index = index   # unique node id: a string

        # public
        self.neighbours = []  # a list of nodes that neighboring the Nodes
        self.orders = {}    # a dictionary of order objects contained in this node. Notice that future order also exist, so we need to choose from it
        self.drivers = {}    # a dictionary of driver objects contained in this node
        self.order_num = 0  # number of existed orders in this node, used to generate unique order id
        self.idle_driver_num = 0  # number of idle drivers in this node
        self.offline_driver_num = 0
        self.order_generator = None

    def clean_node(self):
        self.orders = {}
        self.order_num = 0
        self.drivers = {}
        self.idle_driver_num = 0
        self.offline_driver_num = 0

    def get_node_index(self):
        return self._index

    def get_driver_numbers(self):
        return self.idle_driver_num

    def get_idle_driver_numbers_loop(self):
        temp_idle_driver = 0
        for key, driver in self.drivers.items():
            if driver.onservice is False and driver.online is True:
                temp_idle_driver += 1
        return temp_idle_driver

    def get_off_driver_numbers_loop(self):
        temp_idle_driver = 0
        for key, driver in self.drivers.items():
            if driver.onservice is False and driver.online is False:
                temp_idle_driver += 1
        return temp_idle_driver

    def order_distribution(self, distribution, dis_paras):
        if distribution == 'Poisson':
            self.order_generator = PoissonDistribution(dis_paras)
        elif distribution == 'Gaussian':
            self.order_generator = GaussianDistribution(dis_paras)
        else:
            pass

    def add_order(self, order_id, city_time, destination_grid, duration, price, wait_time=None, begin_co=None, end_co=None):
        current_node_id = self.get_node_index()
        if wait_time is not None:
            self.orders[order_id] = Order(order_id, self,
                                 destination_grid,
                                 city_time,
                                 duration,
                                 price, wait_time)
        else:
            self.orders[order_id] = Order(order_id, self,
                                     destination_grid,
                                     city_time,
                                     duration,
                                     price)
        self.order_num += 1

    def get_active_orders(self, citytime):
        active_orders = []
        for order_id, order in self.orders.items():
            if order.get_begin_time() < citytime:
                active_orders.append(order)
        return active_orders

    def get_idle_drivers(self):
        for driver_id, driver in self.drivers.items():
            assert driver.online is True
            assert driver.onservice is False
        return self.drivers


    def set_neighbours(self, nodes_list):
        self.neighbours = nodes_list

    def set_expired_driver_offline(self, city_time):
        """
            Set the driver passing the offline time to be offline (remove them out of the simulator)
        """
        keys = list(self.drivers.keys())
        for driver_id in keys:
            driver = self.drivers[driver_id]
            if driver.offline_time < city_time:
                self.drivers.pop(driver_id)


    def remove_driver(self, driver_id):
        """ Remove the orders that are dispatched to drivers """
        removed_driver = self.drivers.pop(driver_id, None)
        self.idle_driver_num -= 1
        if removed_driver is None:
            raise ValueError('Nodes.remove_driver: Remove a driver that is not in this node')
        return removed_driver

    def add_driver(self, driver_id, driver):
        self.drivers[driver_id] = driver
        self.idle_driver_num += 1

    def remove_unfinished_order(self, city_time):
        """ Remove the orders that are expired (waiting time is too long) """
        keys = list(self.orders.keys())
        count = 0
        for order_id in keys:
            # order not served
            order = self.orders[order_id]
            if order.get_wait_time()+order.get_begin_time() < city_time:
                self.orders.pop(order_id)
                count += 1
        return count


    def remove_dispatched_order(self, order_id):
        """ Remove the orders that are dispatched to drivers """
        self.orders.pop(order_id)


class Driver(object):
    __slots__ = ("online", "onservice", 'order', 'node', 'city_time', '_driver_id', 'offline_time', 'coordinate', 'pick_up_duration')

    def __init__(self, driver_id, offline_time):
        self.online = True
        self.onservice = False
        self.order = None     # the order this driver is serving
        self.node = None      # the node that contain this driver.
        self.city_time = 0  # track the current system time
        self.offline_time = offline_time  # record the offline time of the driver
        self.coordinate = None
        self.pick_up_duration = 0

        # private
        self._driver_id = driver_id  # unique driver id.

    def print_driver(self):
        print("Driver:", self._driver_id, self.node.get_node_index(), self.online, self.onservice, self.offline_time)

    def set_position(self, node):
        self.node = node

    def get_position(self):
        return self.node

    def set_order_start(self, order):
        self.order = order

    def get_order(self):
        return self.order

    def set_order_finish(self):
        assert self.onservice is True
        self.order = None
        self.onservice = False

    def get_driver_id(self):
        return self._driver_id

    def update_city_time(self):
        self.city_time += 2

    def set_city_time(self, city_time):
        self.city_time = city_time

    def set_offline(self):
        assert self.onservice is False and self.online is True
        self.online = False
        self.node.idle_driver_num -= 1
        self.node.offline_driver_num += 1

    def set_online(self):
        assert self.onservice is False
        self.online = True

    def set_online_for_finish_dispatch(self):
        self.online = True
        assert self.onservice is False

    def take_order(self, order):
        """ take order, driver show up at destination when order is finished
        """
        assert self.online is True
        self.set_order_start(order)
        self.onservice = True
        self.node.idle_driver_num -= 1

    def status_control_eachtime(self, city):

        assert self.city_time == city.city_time
        if self.onservice is True:
            assert self.online is True
            order_end_time = self.order.get_assigned_time() + self.order.get_duration()
            if self.city_time >= order_end_time:
                self.set_position(self.order.get_end_position())
                self.set_order_finish()
                self.node.add_driver(self._driver_id, self)
            elif self.city_time < order_end_time:
                pass
            else:
                raise ValueError('Driver: status_control_eachtime(): order end time less than city time')


class Order(object):
    __slots__ = ('order_id', '_begin_p', '_end_p', '_begin_t',
                 '_t', '_p', '_waiting_time', '_assigned_time',
                 '_begin_coordinate', '_end_coordinate')

    def __init__(self, order_id, begin_position, end_position, begin_time, duration, price, wait_time=600, begin_co=None, end_co=None):
        self.order_id = order_id
        self._begin_p = begin_position  # node
        self._begin_coordinate = begin_co
        self._end_p = end_position      # node
        self._begin_t = begin_time
        self._end_coordinate = end_co
        # self._end_t = end_time
        self._t = duration              # the duration of order.
        self._p = price                 # same as the reward
        self._waiting_time = wait_time  # a order can last for "wait_time" to be taken
        self._assigned_time = -1

    def get_begin_position(self):
        return self._begin_p

    def get_begin_coordinate(self):
        return self._begin_coordinate

    def get_begin_position_id(self):
        return self._begin_p.get_node_index()

    def get_end_position(self):
        return self._end_p

    def get_end_coordinate(self):
        return self._end_coordinate

    def get_begin_time(self):
        return self._begin_t

    def set_assigned_time(self, city_time):
        self._assigned_time = city_time   # 派单了之后才会有这个assigned time，于是初始化的时候没有这个参数

    def get_assigned_time(self):
        return self._assigned_time

    def get_duration(self):
        return self._t

    def get_price(self):
        return self._p

    def get_wait_time(self):
        return self._waiting_time

    def print_order(self):
        print("Order:", self.order_id, self._begin_p.get_node_index(), self._end_p.get_node_index(), self._begin_t, self._t, self._p)

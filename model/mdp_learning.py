from simulator.envs import *
import json
import pickle
from datetime import datetime


class MDP:

    def __init__(self):
        self.gamma = 0.9  # constant
        self.tp = 300  # time period to divide time slot
        with open("data.pkl", "rb") as pk:
            self.data = pickle.load(pk)  # all information

        self.V = {}
        for l in self.data.items():
            t0 = self.calc_tp(l[0])
            g0 = l[1]
            if t0 not in self.V.keys():
                self.V[t0] = {}
            self.V[t0][g0] = 0
            for ll in l[2]:
                t_ = self.calc_tp(ll[0])
                g_ = ll[1]
                if t_ not in self.V.keys():
                    self.V[t_] = {}
                self.V[t_][g_] = 0

    def make_dict(self):
        for l in self.data.items():
            t0 = self.calc_tp(l[0])
            g0 = l[1]
            n = len(l[2])
            for ll in l[2]:
                t_ = self.calc_tp(ll[0])
                g_ = ll[1]
                r = ll[2]
                delta_t = (t_ - t0) // self.tp
                self.V[t0][g0] = self.V[t0][g0] + 1/n * ((self.gamma ** delta_t) * self.V[t_][g_]
                                                         + self.r_gamma(r, delta_t))
        with open('V.pkl', 'wb') as f:
            pickle.dump(self.V, f)

    def r_gamma(self, r, t):
        rst = 0
        for tt in range(0, t, 1):
            rst = rst + (self.gamma ** tt) * r / t
        return rst

    @staticmethod
    def calc_tp(t):
        h = datetime.fromtimestamp(t).hour
        m = datetime.fromtimestamp(t).minute
        s = datetime.fromtimestamp(t).second

        return s + m * 60 + h * 60 * 60




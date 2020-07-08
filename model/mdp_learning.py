from simulator.envs import *
import json
import pickle
from datetime import datetime
import csv
import pandas as pd
from tqdm import tqdm


class MDP:

    def __init__(self):
        self.grids = pd.read_csv('hexagon_grid_table.csv', encoding='utf8', names=["gridID", "lon1", "lat1",
                                                                           "lon2", "lat2", "lon3", "lat3", "lon4",
                                                                           "lat4", "lon5", "lat5", "lon6", "lat6", ])
        self.gamma = 0.9  # constant
        self.tp = 300  # time period to divide time slot
        self.T = 24*60*60  # a large time period
        with open("../../data_processing/mdp_data/data_gps_20161104", "rb") as pk:
            self.data = pickle.load(pk)  # all information
        #  initialize V, all grid and time  slot has initial 0 value
        vv = {}
        for t in range(int(self.T/self.tp)):
            vv[t] = 0
        self.V = {}
        for ID in self.grids['gridID']:
            self.V[ID] = vv.copy()

    def make_dict(self):
        for i in tqdm(range(len(self.data))):
            l = self.data[i]
            t0 = l[0]
            g0 = l[1]
            n = len(l[2])
            for ll in l[2]:
                t_ = ll[0]
                g_ = ll[1]
                r = ll[2]
                if t0 > t_:
                    continue
                delta_t = (t_ - t0)

                self.V[g0][t0] = self.V[g0][t0] + 1/n * ((self.gamma ** delta_t) * self.V[g_][t_]
                                                         + self.r_gamma(r, delta_t))
        with open('V20161104.pkl', 'wb') as f:
            pickle.dump(self.V, f)

    def meanV(self):
        with open('V20161101.pkl', 'rb') as f:
            df1 = pickle.load(f)
        with open('V20161102.pkl', 'rb') as f:
            df2 = pickle.load(f)
        with open('V20161103.pkl', 'rb') as f:
            df3 = pickle.load(f)
        with open('V20161104.pkl', 'rb') as f:
            df4 = pickle.load(f)

        for ID in self.grids['gridID']:
            for t in range(int(self.T/self.tp)):
                self.V[ID][t] = (df1[ID][t]+df2[ID][t]+df3[ID][t]+df4[ID][t])/4
        with open('V0104mean.pkl', 'wb') as f:
            pickle.dump(self.V, f)

    def r_gamma(self, r, t):
        rst = 0
        for tt in range(0, t, 1):
            rst = rst + (self.gamma ** tt) * r / t
        return rst

    #  return the time period
    def calc_tp(self, t):
        h = datetime.fromtimestamp(t).hour
        m = datetime.fromtimestamp(t).minute
        s = datetime.fromtimestamp(t).second

        return int((s + m * 60 + h * 60 * 60) / self.tp)

    def show(self):
        with open('V0104mean.pkl', 'rb') as f:
            df = pickle.load(f)
        print(df['23fae3f1f69dddfa'])


mdp = MDP()
# mdp.make_dict()
mdp.meanV()
mdp.show()


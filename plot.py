import matplotlib.pyplot as plt
import numpy as np


def np_move_avg(a,n,mode="same"):
    a_ = np.array(a)
    for i in range(n//2, len(a)-n//2+1):
        a_[i] = np.sum(a[(i-n//2):(i+n//2+1)])/n
    return a_


with open('out.txt', 'r') as f:
    lines = f.readlines()

reward = [int(line.split()[3]) for line in lines]

fig, axes = plt.subplots(1, 2)
axes[0].plot(reward)
axes[1].plot(np_move_avg(reward, 100))
plt.show()


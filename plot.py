import matplotlib.pyplot as plt
import numpy as np


def np_move_avg(a,n,mode="same"):
    a_ = np.array(a)
    for i in range(n//2, len(a)-n//2+1):
        a_[i] = np.sum(a[(i-n//2):(i+n//2+1)])/n
    return a_


with open('dqn/trial_3', 'r') as f:
    lines = f.readlines()

reward = []
for line in lines:
    if line[0:14] == "Episode reward":
        reward.append(float(line.split()[2]))


fig, axes = plt.subplots(1, 2)
axes[0].plot(reward)
axes[1].plot(np_move_avg(reward, 100))
plt.show()


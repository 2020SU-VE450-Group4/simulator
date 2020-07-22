import matplotlib.pyplot as plt
import numpy as np
import pickle
import random


def np_move_avg(a,n,mode="same"):
    a_ = np.array(a)
    for i in range(n//2, len(a)-n//2+1):
        a_[i] = np.sum(a[(i-n//2):(i+n//2+1)])/n
    return a_


with open('dqn/trial_4', 'r') as f:
    lines = f.readlines()

reward = []
for line in lines:
    if line[0:14] == "Episode reward":
        reward.append(float(line.split()[2]))

with open("tests/sample/group_real_order_20161101.pkl", "rb") as pk:
    real_order_list = pickle.load(pk)

with open("dqn/greedy_1", 'r') as f:
    new_lines = f.readlines()

greedy = []
for l in new_lines:
    if l[0:14] == "Episode reward":
        greedy.append(float(l.split()[2]))

# fig, axes = plt.subplots(1, 2)
# axes[0].8835.22
plt.plot(reward)
plt.plot(range(58, 70), greedy)
value = sum(greedy)/len(greedy)
plt.hlines(value, 0,70)
value2 = sum(reward[-12:])/12
plt.hlines(value2, 0,70)
# axes[1].plot(np_move_avg(reward, 100))
plt.xlabel("Episode")
plt.ylabel("Reward (Accumulated Driver Income)")
plt.show()
print(value, value2, (value2-value)/value*100, (max(reward)-max(greedy))/value*100)

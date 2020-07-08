import tensorflow as tf
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 3000
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount  0.95
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 8000
BATCH_SIZE = 32
BETA = 1.0


class AC(object):
    def __init__(self, a_dim, s_dim,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim * 2 + 2), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.A = tf.placeholder(tf.float32, [None, a_dim], 'a')  # just a placeholder, do not think too much
        self.A_ = tf.placeholder(tf.float32, [None, a_dim], 'a_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.D = tf.placeholder(tf.float32, [None, 1], 'done')
        self.beta = tf.placeholder(tf.float32, [], 'beta')
        self.actor_q = tf.placeholder(tf.float32, None, "actor_q")  # TD_error for actor update
        self.action_index = tf.placeholder(tf.int32, None, "action_index")  # TD_error for actor update

        with tf.variable_scope('Actor'):
            # self.mu = self._build_mu(self.S, self.A, scope='eval')
            self.pi = self._build_a(self.S, self.A, scope='eval')
            # self.mu_ = self._build_mu(self.S_, self.A_, scope='target', reuse=True)
            self.pi_ = self._build_a(self.S_, self.A_, scope='target', reuse=True)
        with tf.variable_scope('Critic'):
            self.q = self._build_c(self.S, self.A, scope='eval')
            q_ = self._build_c(self.S_, self.A_, scope='target', reuse=True)


        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        # a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        # c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        # ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement
        # def ema_getter(getter, name, *args, **kwargs):
        #     return ema.average(getter(name, *args, **kwargs))
        # target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation

        # update critic part
        q_target = self.R + GAMMA * (1 - self.D) * q_
        self.td_error = q_target -self.q  # TODO: it might not be Q' here, might actually be mu
        # td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.critic_loss = tf.reduce_mean(tf.square(self.td_error))
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.critic_loss, var_list=self.ce_params)

        # update actor part
        # log_prob = tf.log(self.acts_prob[0, self.a])   # this line is for A2C
        # self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        # self.actor_loss = - tf.reduce_mean(self.q*self.mu)  # so we can maximize mu*q
        log_policy = tf.log(self.pi[self.action_index, 0])
        self.actor_loss = - tf.reduce_mean(self.actor_q * log_policy)  # so we can maximize log(pi) * q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.actor_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, actions):
        s = np.array([s])
        s = np.repeat(s, len(actions), 0)
        a = np.array(actions)
        a = a.reshape([-1, 1])
        pi = self.sess.run(self.pi, {self.S: s, self.A: a, self.beta: BETA})
        pi = pi.reshape([1,-1])[0]
        a = np.random.choice(actions, p=pi)
        print(pi)
        return a

    def choose_action_max(self, s, actions):
        s = np.array([s])
        s = np.repeat(s, len(actions), 0)
        a = np.array(actions)
        a = a.reshape([-1,1])
        pi = self.sess.run(self.pi, {self.S: s, self.A: a, self.beta: BETA})
        pi = pi.reshape([1, -1])[0]
        idx = np.argmax(pi)
        return actions[idx]

    # def choose_action(self, s, actions):
    #     s = np.array([s])
    #     s = np.repeat(s, len(actions), 0)
    #     a = np.array(actions)
    #     a = a.reshape([-1, 1])
    #     s = np.array(s)
    #     rankings = self.sess.run(self.mu, {self.S: s, self.A: a})
    #     rankings = rankings.reshape([1, -1])[0]
    #     probs = np.exp(rankings/BETA)/ np.sum(np.exp(rankings/BETA))
    #     # print(s, rankings, probs)
    #     a = np.random.choice(actions, p=probs)
    #     print(rankings[-1], probs[-1])
    #     return a
    #
    # def choose_action_max(self, s, actions):
    #     s = np.array([s])
    #     s = np.repeat(s, len(actions), 0)
    #     a = np.array(actions)
    #     a = a.reshape([-1, 1])
    #     s = np.array(s)
    #     rankings = self.sess.run(self.mu_, {self.S_: s, self.A_: a})
    #     rankings = rankings.reshape([1, -1])[0]
    #     idx = np.argmax(rankings)
    #     return actions[idx]

    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim:self.s_dim + self.a_dim+1]
        bs_ = bt[:, -self.s_dim-self.a_dim-1:-self.a_dim-1]
        ba_ = bt[:, -self.a_dim-1:-1]
        bdone = bt[:, -1:]

        self.sess.run(self.ctrain, {self.S: bs, self.A: ba, self.R: br, self.S_: bs_, self.A_: ba_, self.D: bdone})
        self.sess.run(self.atrain, {self.S: bs, self.A: ba})

    def critic_update(self, s, a, r, s_, a_, done):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        a, a_ = np.array([a])[np.newaxis, :], np.array([a_])[np.newaxis, :]
        r, done = np.array([r])[np.newaxis, :], np.array([done])[np.newaxis, :]
        q, _ = self.sess.run([self.q, self.ctrain], {self.S: s, self.A: a, self.R: r, self.S_: s_, self.A_: a_, self.D: done})
        print(q)
        return q

    def actor_update(self, s, actions, a, q):
        action_index = actions.index(a)
        s = np.array([s])
        s = np.repeat(s, len(actions), 0)
        a = np.array(actions)
        a = a.reshape([-1, 1])
        self.sess.run(self.atrain, {self.S: s, self.A: a, self.beta: BETA, self.action_index: action_index, self.actor_q: q})

    def store_transition(self, s, a, r, s_, a_, done):
        transition = np.hstack((s, a, [r], s_, a_, [done]))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_mu(self, s, a, reuse=None, scope=None):
        trainable = True if reuse is None else False
        # MLP with (256, 128, 64)
        with tf.variable_scope(scope):
            s_a = tf.concat([s, a], axis=-1)
            l1 = tf.layers.dense(
                inputs=s_a,
                units=30,  # number of hidden units  256
                activation=tf.nn.relu,
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),  # weights
                name='l1',
                trainable=trainable
            )

            # l2 = tf.layers.dense(
            #     inputs=l1,
            #     units=128,  # number of hidden units
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l2',
            #     trainable=trainable
            # )
            #
            # l3 = tf.layers.dense(
            #     inputs=l2,
            #     units=64,  # number of hidden units
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l3',
            #     trainable=trainable
            # )

            self.mu_out = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=tf.nn.sigmoid,  # sigmoid here, then feed into Boltzmann selector
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),  # weights
                name='mu_out'
            )
            return self.mu_out  # mu(s,a)

    def _build_c(self, s, a, reuse=None, scope=None):
        # MLP with four hidden layers (512, 256, 128, 64）
        trainable = True if reuse is None else False
        with tf.variable_scope(scope):
            s_a = tf.concat([s, a], axis=-1)
            l1 = tf.layers.dense(
                inputs=s_a,
                units=30,  # number of hidden units 512
                activation=tf.nn.relu,
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),  # weights
                name='l1',
                trainable=trainable
            )

            # l2 = tf.layers.dense(
            #     inputs=l1,
            #     units=256,  # number of hidden units
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l2',
            #     trainable=trainable
            # )
            #
            # l3 = tf.layers.dense(
            #     inputs=l2,
            #     units=128,  # number of hidden units
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l3',
            #     trainable=trainable
            # )
            #
            # l4 = tf.layers.dense(
            #     inputs=l3,
            #     units=64,  # number of hidden units
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),  # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l4',
            #     trainable=trainable
            # )

            Q_out = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                # activation=tf.nn.relu,  # TODO: relu here may used to prevent Q value from being negative, but why so?
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),  # weights
                name='Q_out'
            )
            return Q_out  # Q(s,a)

    def _build_a(self, s, a, reuse=None, scope=None):
        trainable = True if reuse is None else False
        # MLP with (256, 128, 64)
        with tf.variable_scope(scope):
            s_a = tf.concat([s, a], axis=-1)
            l1 = tf.layers.dense(
                inputs=s_a,
                units=30,  # number of hidden units  256
                activation=tf.nn.relu,
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),  # weights
                name='l1',
                trainable=trainable
            )
            self.mu_out = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=tf.nn.sigmoid,  # sigmoid here, then feed into Boltzmann selector
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),  # weights
                name='mu_out'
            )
            prob = (tf.exp(self.mu_out)/self.beta) / tf.reduce_sum(tf.exp(self.mu_out)/self.beta)
            return prob




###############################  training  ####################################

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = 1

actions = [0, 1]

algo = AC(a_dim, s_dim)

t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    experience_buffer = None
    for j in range(1000):
        # env.render()
        a = algo.choose_action(s, actions)
        s_, r, done, _ = env.step(a)
        a_ = 0  # default is arbitrary
        # if j == 999 or r>10:
        #     done = True
        if not done:
            a_ = algo.choose_action_max(s_, actions)
        algo.store_transition(s, a, r, s_, a_, done)

        # if algo.pointer > MEMORY_CAPACITY:
        #     BETA *= .99999    # decay the action randomness
        #     if BETA < 0.01:
        #         BETA = 0.01
        #     algo.learn()

        # A2C style
        q = algo.critic_update(s, a, r, s_, a_, done)  # gradient = grad[r + gamma * V(s_) - V(s)]
        algo.actor_update(s, actions, a, q)  # true_gradient = grad[logPi(s,a) * td_error]
        BETA *= .9995

        s = s_
        ep_reward += r
        if done:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % BETA, )
            break


print('Running time: ', time.time() - t1)
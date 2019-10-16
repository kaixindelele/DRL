"""
在DDPG_per的结构基础上改的SAC;
利用spinning-up的基本公式和函数实现;
因为spinning-up的代码用起来不方便,所以还是自己写一个

"""

import tensorflow as tf
import numpy as np
import sys


#####################  hyper parameters  ####################
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement


###############################  DDPG  ####################################

class TD3(object):
    def __init__(self, a_dim, s_dim, a_bound, transition_num=5,
                 batch_size=32, memory_size=100000, per_flag=False,
                 start_steps=10000,
                 act_noise=0.1, target_noise=0.2,
                 noise_clip=0.5, policy_delay=2,):

        # 新加的一些属性:
        self.start_steps = start_steps
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        # 之前的属性
        self.transition_num = transition_num
        self.memory_size = memory_size
        self.per_flag = per_flag
        if per_flag:
            from memory.per_memory import Memory
        else:
            from memory.simple_memory import Memory

        self.memory = Memory(memory_size=memory_size,
                             batch_size=batch_size,
                             transition_num=transition_num,
                             )
        self.batch_size = batch_size

        self.learn_step = 0
        self.per_pointer = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1

        self.sess = tf.Session(config = config)

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.actor_lr = tf.placeholder(tf.float32, shape=[],
                                       name='actor_lr')
        self.critic_lr = tf.placeholder(tf.float32, shape=[],
                                        name='critic_lr')

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.Done = tf.placeholder(tf.float32, [None, 1], 'done')
        self.ISWeights = tf.placeholder(tf.float32, [None, 1],
                                        name='IS_weights')

        with tf.variable_scope('p'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('q1'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            # 终于知道上面注释的意思了:计算q的时候,传入的是ba,然后就直接填充到self.a里面了
            # 但是当没有self.a: ba填充的时候,就会从头开始计算
            q1 = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q1_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        with tf.variable_scope('q2'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q2 = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q2_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # 新加的加噪声的策略平滑
        # Target policy smoothing, by adding clipped noise to target actions
        epsilon = tf.random_normal(tf.shape(a_), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = a_ + epsilon
        self.target_action = tf.clip_by_value(a2, -a_bound, a_bound)

        # networks parameters~
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='p/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='p/target')
        self.c1e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='q1/eval')
        self.c1t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='q1/target')
        self.c2e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='q2/eval')
        self.c2t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='q2/target')

        self.ce_params = self.c1e_params + self.c2e_params

        # hard_replace
        self.hard_replace = [tf.assign(t, e)
                             for t, e in zip(
                self.at_params + self.c1t_params + self.c2t_params,
                self.ae_params + self.c1e_params + self.c2e_params)]

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(
                self.at_params + self.c1t_params + self.c2t_params,
                self.ae_params + self.c1e_params + self.c2e_params)]

        # 梯度更新和优化,核心算法
        with tf.variable_scope('q1', reuse=True):
            q1_target_action = self._build_c(self.S_, self.target_action,
                                             scope='target', trainable=False)
        with tf.variable_scope('q2', reuse=True):
            q2_target_action = self._build_c(self.S_, self.target_action,
                                             scope='target', trainable=False)

        q_min = tf.minimum(q1_target_action, q2_target_action)
        if self.transition_num == 5:
            # q_target = self.R + GAMMA * q_ * (1 - self.Done)
            q_target = tf.stop_gradient(self.R + GAMMA * q_min * (1 - self.Done))

        pi_loss = -tf.reduce_mean(q1)
        q1_loss = tf.reduce_mean((q1 - q_target) ** 2)
        q2_loss = tf.reduce_mean((q2 - q_target) ** 2)
        # 为啥这里的loss是加起来的?
        q_loss = q1_loss + q2_loss
        self.q_loss = q_loss

        self.ctrain = tf.train.AdamOptimizer(
            self.critic_lr).minimize(q_loss,
                                     var_list=self.ce_params)

        self.a_loss = pi_loss
        self.atrain = tf.train.AdamOptimizer(
            self.actor_lr).minimize(self.a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.hard_replace)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def get_action(self, s, noise_scale):
        a = self.sess.run(self.a, feed_dict={self.S: s.reshape(1, -1)})[0]
        a += noise_scale * np.random.randn(self.a_dim)
        return np.clip(a, -self.a_bound, self.a_bound)

    def store_transition(self, transition):
        self.memory.store(transition)
        if self.per_flag:
            self.per_pointer = self.memory.tree.data_pointer
        else:
            self.per_pointer = self.memory.memory_num

    def learn(self, actor_lr_input, critic_lr_input,
              output_loss_flag=False):
        # soft target replacement
        self.sess.run(self.soft_replace)
        self.learn_step += 1
        a_loss = 0.0
        c_loss = 0.0
        if self.per_flag:
            tree_idx, batch_memory, ISWeights = self.memory.sample()

            if self.transition_num == 4:
                batch_states, batch_actions, batch_rewards, batch_states_ = [], [], [], []
                for i in range(self.batch_size):
                    batch_states.append(batch_memory[i][0])
                    batch_actions.append(batch_memory[i][1])
                    batch_rewards.append(batch_memory[i][2])
                    batch_states_.append(batch_memory[i][3])

                bs = np.array(batch_states)
                ba = np.array(batch_actions)
                batch_rewards = np.array(batch_rewards)
                bs_ = np.array(batch_states_)
                br = batch_rewards[:, np.newaxis]

                _, a_loss = self.sess.run([self.atrain, self.a_loss], {self.S: bs, self.actor_lr: actor_lr_input})
                _, abs_errors, cost = self.sess.run([self.ctrain, self.abs_errors, self.c_loss],
                                                    {self.S: bs, self.a: ba, self.R: br, self.S_: bs_,
                                                     self.critic_lr: critic_lr_input,
                                                     self.ISWeights: ISWeights})

                self.memory.batch_update(tree_idx, abs_errors)  # update priority
                return a_loss, cost
            if self.transition_num == 5:
                batch_states, batch_actions, batch_rewards, batch_states_, batch_dones = [], [], [], [], []
                for i in range(self.batch_size):
                    batch_states.append(batch_memory[i][0])
                    batch_actions.append(batch_memory[i][1])
                    batch_rewards.append(batch_memory[i][2])
                    batch_states_.append(batch_memory[i][3])
                    batch_dones.append(batch_memory[i][4])

                bs = np.array(batch_states)
                ba = np.array(batch_actions)
                batch_rewards = np.array(batch_rewards)
                bs_ = np.array(batch_states_)
                br = batch_rewards[:, np.newaxis]
                batch_dones = np.array(batch_dones)
                bd = batch_dones[:, np.newaxis]

                _, a_loss = self.sess.run([self.atrain, self.a_loss], {self.S: bs, self.actor_lr: actor_lr_input})
                _, abs_errors, cost = self.sess.run([self.ctrain, self.abs_errors, self.c_loss],
                                                    {self.S: bs, self.a: ba, self.R: br, self.S_: bs_,
                                                     self.Done: bd,
                                                     self.critic_lr: critic_lr_input,
                                                     self.ISWeights: ISWeights})

                self.memory.batch_update(tree_idx, abs_errors)  # update priority
                return a_loss, cost

        else:
            # 加上terminal信息
            if self.transition_num == 5:
                bs, ba, br, bs_, bt = self.memory.sample()
                if output_loss_flag:
                    if self.learn_step % self.policy_delay == 0:
                        _, a_loss = self.sess.run([self.atrain, self.a_loss],
                                                  {self.S: bs,
                                                   self.actor_lr: actor_lr_input})
                    _, c_loss = self.sess.run([self.ctrain, self.q_loss],
                                              {self.S: bs, self.a: ba,
                                               self.R: br, self.S_: bs_,
                                               self.Done: bt,
                                               self.critic_lr: critic_lr_input})
                    return a_loss, c_loss
                else:
                    if self.learn_step % self.policy_delay == 0:
                        self.sess.run(self.atrain, {self.S: bs,
                                                    self.actor_lr: actor_lr_input})
                    self.sess.run(self.ctrain, {self.S: bs, self.a: ba,
                                                self.R: br, self.S_: bs_,
                                                self.Done: bt,
                                                self.critic_lr: critic_lr_input})

            if self.transition_num == 4:
                bs, ba, br, bs_ = self.memory.sample()
                if output_loss_flag:
                    if self.learn_step % self.policy_delay == 0:
                        _, a_loss = self.sess.run([self.atrain, self.a_loss],
                                                  {self.S: bs,
                                                   self.actor_lr: actor_lr_input})
                    _, c_loss = self.sess.run([self.ctrain, self.q_loss],
                                              {self.S: bs, self.a: ba,
                                               self.R: br, self.S_: bs_,
                                               self.critic_lr: critic_lr_input})
                    return a_loss, c_loss
                else:
                    if self.learn_step % self.policy_delay == 0:
                        self.sess.run(self.atrain, {self.S: bs,
                                                    self.actor_lr: actor_lr_input})
                    self.sess.run(self.ctrain, {self.S: bs, self.a: ba,
                                                self.R: br, self.S_: bs_,
                                                self.critic_lr: critic_lr_input})

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            new_actor_layer = tf.layers.dense(net, 200, activation=tf.nn.relu, name='new_actor_layer', trainable=trainable)
            a = tf.layers.dense(new_actor_layer, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            new_critic_layer = tf.layers.dense(net, 300, activation=tf.nn.relu, name='new_critic_layer',
                                               trainable=trainable)
            return tf.layers.dense(new_critic_layer, 1, trainable=trainable)  # Q(s,a)

    def load_step_network(self, saver, load_path):
        checkpoint = tf.train.get_checkpoint_state(load_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, tf.train.latest_checkpoint(load_path))
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            self.learn_step = int(checkpoint.model_checkpoint_path.split('-')[-1])
        else:
            print("Could not find old network weights")

    def save_step_network(self, time_step, saver, save_path):
        saver.save(self.sess, save_path + 'network', global_step=time_step,
                   write_meta_graph=False)

    def load_simple_network(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(path))
        print("restore model successful")

    def save_simple_network(self, save_path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path=save_path + "/params", write_meta_graph=False)

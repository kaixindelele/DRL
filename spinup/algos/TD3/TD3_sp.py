import numpy as np
import tensorflow as tf
import gym
import time
import sys
sys.path.append("../")
from td3 import core
from td3.core import get_vars, actor_critic


#####################  hyper parameters  ####################
GAMMA = 0.99     # reward discount
TAU = 0.005      # soft replacement
polyak = 0.995

"""

TD3 (Twin Delayed DDPG)

"""


class TD3(object):
    def __init__(self, a_dim, s_dim, a_bound, transition_num=5,
                 batch_size=32, memory_size=100000, per_flag=False,
                 start_steps=10000, ac_kwargs=dict(),
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

        from memory.simple_memory import Memory

        self.memory = Memory(memory_size=memory_size,
                             batch_size=batch_size,
                             transition_num=transition_num,
                             )
        self.batch_size = batch_size

        self.learn_step = 0
        self.per_pointer = 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,

        # 下面的网络结构定义全部用spinningup!
        # 创建placeholder
        self.actor_lr = tf.placeholder(tf.float32, shape=[],
                                       name='actor_lr')
        self.critic_lr = tf.placeholder(tf.float32, shape=[],
                                        name='critic_lr')

        self.S, self.A, self.S_, self.R, self.Done = core.placeholders(
                                                      self.s_dim,
                                                      self.a_dim,
                                                      self.s_dim,
                                                      None,
                                                      None)
        ac_kwargs['action_space'] = self.a_bound
        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.pi, self.q1, self.q2, self.q1_pi = core.actor_critic(
                                                          self.S,
                                                          self.A,
                                                          **ac_kwargs)

        # Target policy network
        with tf.variable_scope('target'):
            self.pi_targ, _, _, _ = actor_critic(self.S_, self.A,
                                                 **ac_kwargs)

        # Target Q networks

        with tf.variable_scope('target', reuse=True):
            # Target policy smoothing, by adding clipped noise to target actions
            # epsilon = tf.random_normal(tf.shape(self.pi_targ),
            #                            stddev=target_noise)
            # epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            # a2 = self.pi_targ + epsilon
            # a2 = tf.clip_by_value(a2, -self.a_bound, self.a_bound)
            #
            # # Target Q-values, using action from target policy
            # _, self.q1_targ, self.q2_targ, _ = actor_critic(self.S_, a2,
            #                                                 **ac_kwargs)
            _, self.q1_targ, self.q2_targ, _ = actor_critic(self.S_,
                                                            self.pi_targ,
                                                            **ac_kwargs)

        # Count variables
        var_counts = tuple(core.count_vars(scope)
                           for scope in ['main/pi',
                                         'main/q1',
                                         'main/q2',
                                         'main'])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n' % var_counts)

        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(self.q1_targ, self.q2_targ)
        backup = tf.stop_gradient(self.R + GAMMA * (1 - self.Done) * min_q_targ)

        # TD3 losses
        pi_loss = -tf.reduce_mean(self.q1_pi)
        self.pi_loss = pi_loss
        q1_loss = tf.reduce_mean((self.q1 - backup) ** 2)
        q2_loss = tf.reduce_mean((self.q2 - backup) ** 2)
        # 为啥这里的loss是加起来的?
        q_loss = q1_loss + q2_loss
        self.q_loss = q_loss

        # Separate train ops for pi, q
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        [print("get_vars('main/pi'):", var) for var in get_vars('main/pi')]
        self.train_pi_op = pi_optimizer.minimize(pi_loss,
                                                 var_list=get_vars('main/pi'))
        [print("get_vars('main/q'):", var) for var in get_vars('main/q')]
        self.train_q_op = q_optimizer.minimize(q_loss,
                                               var_list=get_vars('main/q'))

        # Polyak averaging for target variables
        self.target_update = tf.group(
            [tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
             for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # self.target_update = tf.group(
        #     [tf.assign(v_targ, (1-TAU) * v_targ + TAU * v_main)
        #      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # Initializing targets to match main variables
        target_init = tf.group(
            [tf.assign(v_targ, v_main)
             for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(target_init)

    def choose_action(self, s):
        return self.sess.run(self.pi, {self.S: s[np.newaxis, :]})[0]

    def get_action(self, s, noise_scale):
        a = self.sess.run(self.pi, feed_dict={self.S: s.reshape(1, -1)})[0]
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
        self.sess.run(self.target_update)
        self.learn_step += 1
        pi_loss, q_loss = 0.0, 0.0
        q1, q2 = 0.0, 0.0

        bs, ba, br, bs_, bt = self.memory.sample()

        feed_dict = {
                    self.S: bs,
                    self.A: ba,
                    self.R: br,
                    self.S_: bs_,
                    self.Done: bt,
                    self.actor_lr: actor_lr_input,
                    self.critic_lr: critic_lr_input,
                    }

        q_step_ops = [self.q_loss, self.q1, self.q2, self.train_q_op]
        pi_step_ops = [self.train_pi_op, self.pi_loss]

        if self.learn_step % self.policy_delay == 0:
            _, pi_loss = self.sess.run(pi_step_ops, feed_dict=feed_dict)

        _, q1, q2, q_loss = self.sess.run(q_step_ops, feed_dict=feed_dict)
        return pi_loss, q1, q2, q_loss

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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    args = parser.parse_args()

    # td3(lambda: gym.make(args.env), actor_critic=core.mlp_actor_critic,
    #     ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
    #     gamma=args.gamma, seed=args.seed, epochs=args.epochs,
    #     logger_kwargs=logger_kwargs)




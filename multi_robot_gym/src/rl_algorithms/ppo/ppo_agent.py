import numpy as np
import tensorflow as tf
import gym
import time
import core
import rospy

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        x = np.array(self.adv_buf,dtype=np.float32)
        length = len(x)
        sum = np.sum(x)
        adv_mean = sum / length
        adv_std = np.sqrt(np.sum((x-adv_mean)**2)/length)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]


"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""


class PPOAgent:
    def __init__(self, observation_space,action_space):
        obs_dim = observation_space.shape
        act_dim = action_space.shape

        # Share information about action space with policy architecture
        ac_kwargs = dict()
        ac_kwargs['action_space'] = action_space
        #ac_kwargs['output_activation'] = tf.tanh

        # Inputs to computation graph
        self.x_ph,  self.a_ph = core.placeholders_from_spaces(observation_space, action_space)
        self.adv_ph, self.ret_ph, self.logp_old_ph = core.placeholders(None, None, None)

        # Main outputs from computation graph
        self.pi, self.logp, self.logp_pi, self.v = core.mlp_actor_critic(self.x_ph, self.a_ph, output_activation=tf.tanh,**ac_kwargs)

        # Need all placeholders in *this* order later (to zip with data from buffer)
        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]

        # Every step, get: action, value, and logprob
        self.get_action_ops = [self.pi, self.v, self.logp_pi]

        # Experience buffer
        steps_per_epoch = 1000
        self.local_steps_per_epoch = steps_per_epoch
        gamma = 0.99
        lam = 0.97
        self.buf = PPOBuffer(obs_dim, act_dim, self.local_steps_per_epoch, gamma, lam)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
        print var_counts

        # PPO objectives
        clip_ratio = 0.2
        ratio = tf.exp(self.logp - self.logp_old_ph)  # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(self.adv_ph > 0, (1 + clip_ratio) * self.adv_ph, (1 - clip_ratio) * self.adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv))
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v) ** 2)

        # Info (useful to watch during learning)
        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)  # a sample estimate for KL-divergence, easy to compute
        self.approx_ent = tf.reduce_mean(-self.logp)  # a sample estimate for entropy, also easy to compute
        self.clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
        self.clipfrac = tf.reduce_mean(tf.cast(self.clipped, tf.float32))
        pi_lr = 3e-4
        vf_lr = 1e-3
        pi_optimizer = tf.train.AdadeltaOptimizer(learning_rate=pi_lr)
        vf_optimizer = tf.train.AdadeltaOptimizer(learning_rate=vf_lr)
        self.train_pi = pi_optimizer.minimize(self.pi_loss)
        self.train_v = vf_optimizer.minimize(self.v_loss)
        self.train_pi_iters = 80
        self.train_v_iters = 80
        self.target_kl = 0.01

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        inputs = {k:v for k,v in zip(self.all_phs, self.buf.get())}
        pi_l_old, v_l_old, ent = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)
        print pi_l_old
        print v_l_old

        # Training
        for i in range(self.train_pi_iters):
            _, kl = self.sess.run([self.train_pi, self.approx_kl], feed_dict=inputs)
            # kl = mpi_avg(kl)
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step %d due to reaching max kl.' % i)
                break
        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed_dict=inputs)

        # debug info from update
        pi_l_new, v_l_new, kl, cf = self.sess.run([self.pi_loss, self.v_loss, self.approx_kl, self.clipfrac], feed_dict=inputs)

    def get_action(self, observation):
        a, v_t, logp_t = self.sess.run(self.get_action_ops, feed_dict={self.x_ph: observation.reshape(1, -1)})
        #print observation
        #print observation.reshape(1,-1)
        #print a
        #print v_t
        #print logp_t
        return a, v_t, logp_t

    def add_experience(self, o, a, r, v_t, logp_t):
        self.buf.store(o, a, r, v_t, logp_t)

    def finish_path(self, r, done, o):
        last_val = r if done else self.sess.run(self.v, feed_dict={self.x_ph: o.reshape(1, -1)})
        self.buf.finish_path(last_val)


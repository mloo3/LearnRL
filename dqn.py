"""
-Experience Replay
-Target Network
-Clipping Rewards
-Skipping Frames
"""
import gym
from buffers import ReplayBuffer
from networks import Mlpnn

import random
import numpy as np

import tensorflow as tf
from tensorflow import keras as kr


class DQN:
    def __init__(self, env, buffer_size=10000, gamma=0.99,
                init_eps=0.9, final_eps=0.1, exploration_fraction=0.1,
                learning_starts=1000, batch_size=32, learning_rate=0.01,
                seed=None):
        self.env = env
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.init_eps = init_eps
        self.final_eps = final_eps
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

        obs = self.env.reset()

        self.obs_shape = obs.shape
        self.act_shape = self.env.action_space.n # this might only work for gym.Discrete

        self.model = None



    def learn(self, timesteps=10000, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        self.eps_range = self._eps_range(timesteps)
        replay_buffer = ReplayBuffer(self.buffer_size)

        self._init_model()

        cur_step = 0
        for step in range(timesteps):
            obs = self.env.reset()
            done = False
            while not done:
                cur_eps = next(self.eps_range)
                action = self._select_action(np.array(obs), cur_eps)

                new_obs, rewards, done, info = env.step(action)
                if done:
                    new_obs = None
                replay_buffer.add(obs, action, rewards, new_obs)

                obs = new_obs

                # learn gradient
                if cur_step > self.learning_starts:
                    if len(replay_buffer.buffer) < self.batch_size: # buffer too small
                        continue
                    samples = replay_buffer.sample(self.batch_size)
                    obs_batch, actions_batch, rewards_batch, new_obs_batch = samples

                    # q_value = self._predictQValue()




                cur_step += 1





    def _eps_range(self, timesteps):
        end_step = int(timesteps * self.exploration_fraction)
        step_size = (self.init_eps - self.final_eps) / end_step
        cur_step = 0
        cur_eps = self.init_eps
        while cur_step < end_step + 1:
            yield cur_eps
            cur_eps -= step_size
            cur_step += 1

    def _select_action(self, obs, epsilon):
        r = random.uniform(0,1)
        if r > epsilon:
            # TODO
            # action = network
            action = 0
        else:
            action = np.array([[random.randrange(obs.action_space.n)]])

        return action

    # def _init_model(self):
    #
    #     self.model = Mlpnn([], self.obs_shape, self.act_shape)
    #     optimizer = tf.keras.optimizers.Adam(self.learning_rate)
    #
    #     self.model.compile(optimizer, loss='mse') # maybe different loss?

    # def _predictQValue(self):





if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    obs = env.reset()
    dqn = DQN(env, init_eps=1, final_eps=0.1, exploration_fraction=0.1)

    dqn._init_model()






"""
-Experience Replay
-Target Network
-Clipping Rewards
-Skipping Frames
"""
import gym
from buffer import ReplayBuffer

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



class DQN:
    def __init__(self, env, buffer_size=10000, gamma=0.99, init_eps=0.9, final_eps=0.1, exploration_fraction=0.1, learning_starts=1000):
        self.env = env
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.init_eps = init_eps
        self.final_eps = final_eps
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts



    def learn(self, timesteps=10000, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(random_seed)

        self.eps_range = self._eps_range(timesteps)
        replay_buffer = ReplayBuffer(self.buffer_size)
        for step in range(timesteps):
            obs = self.env.reset()
            done = False
            while not done:
                cur_eps = next(self.eps_range)
                action = self._select_action(np.array(obs), cur_eps)

                new_obs, rewards, dones, info = env.step(action)
                if dones:
                    new_obs = None
                replay_buffer.add(obs, action, rewards, new_obs)

                obs = new_obs




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
            action = torch.tensor([[random.randrange(obs.action_space.n)]])

        return action





if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    obs = env.reset()
    dqn = DQN(env, init_eps=1, final_eps=0.1, exploration_fraction=0.1)
    test = dqn._eps_range(100)

    # dqn.learn()






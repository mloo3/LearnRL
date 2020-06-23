"""
-Experience Replay (done)
-Target Network (done)
-Clipping Rewards
-Skipping Frames (no)
"""
import gym
from buffers import ReplayBuffer
from networks import MlpNN

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQN:
    def __init__(self, env, buffer_size=10000, gamma=0.99,
                init_eps=0.9, final_eps=0.1, exploration_fraction=0.1,
                learning_starts=1000, batch_size=32, learning_rate=0.01,
                target_network_update_freq=500,
                seed=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.env = env
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.init_eps = init_eps
        self.final_eps = final_eps
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_network_update_freq = target_network_update_freq
        self.seed = seed

        obs = self.env.reset()

        self.obs_shape = obs.shape
        self.act_shape = self.env.action_space.n # this might only work for gym.Discrete

        self.step_model = None
        self.target_model = None
        self.optim = None



    def learn(self, timesteps=10000, verbose=0, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.eps_range = self._eps_range(timesteps)
        replay_buffer = ReplayBuffer(self.buffer_size)

        self._init_model()

        obs = self.env.reset()
        for step in range(timesteps):
            # while not done:
            cur_eps = next(self.eps_range, None)
            if cur_eps is None:
                cur_eps = self.final_eps

            action = self._select_action(obs, cur_eps)

            new_obs, rewards, done, info = self.env.step(action)
            if done:
                new_obs = [np.nan] * self.obs_shape[0] # hacky way to keep dimensions correct
            replay_buffer.add(obs, action, rewards, new_obs)

            obs = new_obs

            # learn gradient
            if step > self.learning_starts:
                if len(replay_buffer.buffer) < self.batch_size: # buffer too small
                    continue
                samples = replay_buffer.sample(self.batch_size, self.device)
                obs_batch, actions_batch, rewards_batch, new_obs_batch = samples

                predicted_q_values = self._predictQValue(self.step_model, obs_batch, actions_batch)
                ys = self._expectedLabels(self.target_model, new_obs_batch, rewards_batch)

                loss = F.smooth_l1_loss(predicted_q_values, ys)

                self.optim.zero_grad()
                loss.backward()
                for i in self.step_model.parameters():
                    i.grad.clamp_(min=-1, max=1) # exploding gradient
                    # i.grad.clamp_(min=-10, max=10) # exploding gradient
                self.optim.step()

                # update target
                if step % self.target_network_update_freq == 0:
                    self.target_model.load_state_dict(self.step_model.state_dict())

            if done:
                obs = self.env.reset()
            if verbose == 1:
                if step % (timesteps * 0.1) == 0:
                    perc = int(step / (timesteps * 0.1))
                    print(f"At step {step}")
                    print(f"{perc}% done")

    def predict(self, obs):
        obs = obs.reshape(1, obs.shape[0])
        obs = torch.from_numpy(obs).float().to(self.device)
        q_value = self.step_model(obs)
        action = torch.argmax(q_value)
        action = int(action.cpu())
        return action



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
            obs = obs.reshape(1,obs.shape[0])
            obs = torch.from_numpy(obs).float().to(self.device)
            action = self.step_model(obs) # returns np array
            action = torch.argmax(action)
            action = int(action.cpu())
        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def _init_model(self):

        self.step_model = MlpNN([], self.obs_shape[0], self.act_shape).to(self.device)
        self.target_model = MlpNN([], self.obs_shape[0], self.act_shape).to(self.device)

        self.optim = optim.Adam(self.step_model.parameters(), lr=self.learning_rate)

    def _predictQValue(self, model, obs, acts):
        q_values = model(obs)
        q_values = torch.gather(q_values, 1, acts)
        return q_values

    def _expectedLabels(self, model, new_obs, rewards):
        q_values = torch.zeros(self.batch_size, 1).float().to(self.device)
        mask = ~torch.isnan(torch.sum(new_obs, axis=1))
        q_value = model(new_obs[mask])
        max_q_value = torch.max(q_value, axis=1).values
        max_q_value = torch.reshape(max_q_value, (max_q_value.shape[0], 1))

        q_values[mask] = max_q_value
        ys = rewards + (q_values * self.gamma)
        return ys









if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    obs = env.reset()
    dqn = DQN(env, init_eps=1, final_eps=0.1, exploration_fraction=0.1, learning_starts=500, target_network_update_freq=100)
    dqn.learn(1000)
    dqn.predict(obs)
    # test = np.zeros((32,2))
    # test[:,0] = -2
    # test[:,1] = -1
    # test2 = np.random.randint(0,1, size=(32,))
    # print(test[np.arange(test.shape[0]),test2].shape)
    # print(tf.gather(test,test2,axis=0))
    # dqn._predictAction(test)
    # dqn.learn(2000)







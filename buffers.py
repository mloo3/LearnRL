import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, size=100):
        self.size = size
        self.buffer = []
        self.next_idx = 0

    def add(self, obs, action, reward, obs_prime):
        experience = (obs, action, reward, obs_prime)
        if len(self.buffer) < self.size:
            self.buffer.append(experience)
        else:
            self.buffer[self.next_idx] = experience
        self.next_idx = (self.next_idx + 1) % self.size
    def sample(self, batch_size, device):
        batch_ixs = np.random.randint(0,len(self.buffer) - 1, size=batch_size)

        o,a,r,o_p = self.buffer[0]
        obs_batch = []
        actions_batch = []
        rewards_batch = []
        new_obs_batch = []

        for ix in batch_ixs:
            obs_batch.append(self.buffer[ix][0])
            actions_batch.append(self.buffer[ix][1])
            rewards_batch.append(self.buffer[ix][2])
            new_obs_batch.append(self.buffer[ix][3])

        # may want to switch this to tensors later on
        obs_batch = torch.Tensor(obs_batch).to(device)

        actions_batch = torch.Tensor(actions_batch).long().to(device)
        actions_batch = torch.reshape(actions_batch, (actions_batch.shape[0], 1))

        rewards_batch = torch.Tensor(rewards_batch).to(device)
        rewards_batch = torch.reshape(rewards_batch, (rewards_batch.shape[0], 1))

        new_obs_batch = torch.Tensor(new_obs_batch).to(device)

        return obs_batch, actions_batch, rewards_batch, new_obs_batch



if __name__ == "__main__":
    test = ReplayBuffer(10)
    for i in range(50):
        test.add(i,i,i,i)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    a = test.sample(5,device)
    print(a)



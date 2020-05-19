import numpy as np

class ReplayBuffer:
    def __init__(self, size=100):
        self.size = size
        self.buffer = []
        self.next_idx = 0

    def add(self, obs, action, reward, obs_prime):
        experience = (obs, action, reward, obs_prime)
        if len(self.buffer) < self.next_idx:
            self.buffer.append(experience)
        else:
            self.buffer[self.next_idx] = experience
        self.next_idx = (self.next_idx + 1) % self.size



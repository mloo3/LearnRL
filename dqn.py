"""
-Experience Replay
-Target Network
-Clipping Rewards
-Skipping Frames
"""
from buffer import ReplayBuffer
class DQN:
    def __init__(self, buffer_size=100):
        self.buffer_size = 100


    def learn(self, episodes=10, ):
        replayBuffer = ReplayBuffer(self.buffer_size)

        # for episode in range(episodes):




import torch
import torch.nn as nn
import torch.nn.functional as F




class MlpNN(nn.Module):
    def __init__(self, arch, obs_shape, act_shape, ):
        #TODO: make custom arch through layers work
        # add batch norm?https://stats.stackexchange.com/questions/304755/pros-and-cons-of-weight-normalization-vs-batch-normalization
        super(MlpNN, self).__init__()
        self.fc1 = nn.Linear(obs_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_shape)


    def forward(self, x):
        score = F.relu(self.fc1(x))
        score = F.relu(self.fc2(score))
        score = self.fc3(score)

        return score







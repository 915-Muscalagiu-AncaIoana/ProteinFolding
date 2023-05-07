import random

from torch import nn
import torch.nn.functional as F

from replay_memory import device


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class FCN_QNet(nn.Module):
    """
    action value function, Q(S, a)
    produce the actions in parallel as output vector,
    and choose the max
    """
    def __init__(self, insize, outsize):
        """
        insize ==> input size
            == size of the observation space
        outsize ==> output size
            == number of actions
        """
        super(FCN_QNet, self).__init__()
        self.fc1 = nn.Linear(insize, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, outsize)

    def forward(self, x):
        """
        standard 3-layer fully connected NN
        """
        x = x.to(device)  # for CUDA
        # print("input x.size() = ", x.size())
        #x = x.view(x.size(0),-1)
        # may encounter view memory error
        # RuntimeError: view size is not compatible with input tensor's size and stride
        # (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        x = x.reshape(x.size(0),-1)
        # print("after x.view ---> input x.size() = ", x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            # print("coin < epsilon", coin, epsilon)
            # for 3actionStateEnv use [0,1,2]
            value = random.randint(1, 4)
            return value
        else:
            # print("exploit")
            out = self.forward(obs)

            return out.argmax().item() + 1

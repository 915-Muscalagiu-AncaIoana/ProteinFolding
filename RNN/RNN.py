# Recurrent neural network (many-to-one)
import random

import torch
from torch import nn

sequence_length = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN_LSTM_onlyLastHidden(nn.Module):
    """
    LSTM version that just uses the information from the last hidden state
    since the last hidden state has information from all previous states
    basis for BiDirectional LSTM
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM_onlyLastHidden, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # change basic RNN to LSTM
        # num_layers Default: 1
        # bias Default: True
        # batch_first Default: False
        # dropout Default: 0
        # bidirectional Default: False
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # remove the sequence_length
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Get data to cuda if possible
        x = x.to(device)
        # print("input x.size() = ", x.size())
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # LSTM needs a separate cell state (LSTM needs both hidden and cell state)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        # need to give LSTM both hidden and cell state (h0, c0)
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        # no need to reshape the out or concat
        # out is going to take all mini-batches at the same time + last layer + all features
        out = self.fc(out[:, -1, :])
        # print("forward out = ", out)
        return out

    def sample_action(self, obs, epsilon):
        # print("Sample Action called+++")
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            # print("coin < epsilon", coin, epsilon)
            # for 3actionStateEnv use [0,1,2]
            explore_action = random.randint(0, 5)
            # print("explore_action = ", explore_action)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()




class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # bidrectional=True for BiLSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        # hidden_size needs to expand both directions, *2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Get data to cuda if possible
        x = x.to(device)
        # print("input x.size() = ", x.size())
        # concat both directions, so need to times 2
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # the _ is the (hidden_state, cell_state), but not used
        out, _ = self.lstm(x, (h0, c0))
        # only take the last hidden state to send to the linear layer
        out = self.fc(out[:, -1, :])

        return out

    def sample_action(self, obs, epsilon):
        # print("Sample Action called+++")
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            # print("coin < epsilon", coin, epsilon)
            # for 3actionStateEnv use [0,1,2]
            explore_action = random.randint(0,5)
            # print("explore_action = ", explore_action)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()

class Bi_RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=11, num_layers=2, rnn_type='LSTM'):
        super(Bi_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        #Define the initial linear hidden layer
        self.init_linear = nn.Linear(self.input_dim, self.input_dim)

        # Define the LSTM layer
        self.lstm = eval('nn.' + rnn_type)(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * 2, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))


    def forward(self, input):
        #Forward pass through initial hidden layer
        linear_input = self.init_linear(input)

        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size ,hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (batch_size, num_layers, hidden_dim).
        lstm_out, self.hidden = self.lstm(linear_input)

        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out)
        return y_pred
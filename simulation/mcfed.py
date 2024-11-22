import torch
import torch.nn as nn
import torch.nn.functional as F


class AECF(nn.Module):
    def __init__(self, num_items, hidden_dim, Y):
        super(AECF, self).__init__()

        self.Y = Y
        self.encoder = nn.Linear(num_items, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, r_i):
        x_i = F.relu(self.encoder(r_i))
        Y = self.Y.to(r_i.device)
        r_i_reconstructed = torch.matmul(x_i, Y.T)
        # perform sigmoid to get the values between 0 and 1
        r_i_reconstructed = torch.sigmoid(r_i_reconstructed)
        return r_i_reconstructed

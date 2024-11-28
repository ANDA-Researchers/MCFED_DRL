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


class BaseModel(nn.Module):
    def __init__(self, hidden_dim, num_items):
        super(BaseModel, self).__init__()
        self.user_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, item_id):
        user_embedding = self.user_embedding.repeat(item_id.shape[0], 1)
        item_embedding = self.item_embedding(item_id)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x).squeeze(-1)
        return x

    def predict(self, item_id):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            return self.forward(item_id)

    def get_user_embedding(self):
        return self.user_embedding.detach().cpu().numpy()

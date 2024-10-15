import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    """Simple Deep Neural Network for next item prediction
    Input:
        - item embeddings: (batch_size, embedding_dim)
    Output:
        - probability that the item is the next item: (batch_size, 1)"""

    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x)


class BiLSTM(nn.Module):
    """BiLSTM for next item prediction
    Input:
        - item embeddings: (batch_size, sequence_length, embedding_dim)
    Output:
        - next item prediction: (batch_size, num_items)"""

    def __init__(self, embedding_dim, hidden_dim, num_items):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_items = num_items

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2item = nn.Linear(hidden_dim * 2, num_items)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        item_space = self.hidden2item(lstm_out)
        item_scores = F.log_softmax(item_space, dim=2)
        return item_scores

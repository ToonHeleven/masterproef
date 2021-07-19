import torch.nn as nn
from torch import zeros


class LSTM(nn.Module):
    def __init__(self, max_apps, n_lstm, src_dim=4, bidirectional=True):
        super().__init__()
        self.src_embedding = nn.Embedding(num_embeddings=max_apps, embedding_dim=src_dim)
        self.lstm = nn.LSTM(input_size=src_dim, hidden_size=n_lstm, bidirectional=bidirectional, batch_first=True)
        self.fc1 = nn.Linear(in_features=n_lstm+bidirectional*n_lstm, out_features=max_apps)

        self.__bidirectional = bidirectional
        self.__n_lstm=n_lstm

    def forward(self, x, h):
        h0, c0 = h
        x = self.src_embedding(x)
        x, h = self.lstm(x, (h0.detach(), c0.detach()))
        x = x[:, -1]
        x = self.fc1(x)
        return x, h

    def init_hidden(self, batch_size):
        return zeros(1+self.__bidirectional*1, batch_size, self.__n_lstm), zeros(1+self.__bidirectional*1, batch_size, self.__n_lstm)
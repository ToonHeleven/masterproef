import torch.nn as nn
from torch import zeros

class BiGRU(nn.Module):
    def __init__(self, max_apps, n_gru, src_dim=4, bidirectional=True):
        super().__init__()
        self.src_embedding = nn.Embedding(num_embeddings=max_apps, embedding_dim=src_dim)
        self.bigru = nn.GRU(input_size=src_dim, hidden_size=n_gru, bidirectional=bidirectional, batch_first=True)
        self.fc1 = nn.Linear(in_features=n_gru+bidirectional*n_gru, out_features=max_apps)

        self.__bidirectional=bidirectional
        self.__n_gru=n_gru

    def forward(self, x, h):
        x = self.src_embedding(x)
        x, h = self.bigru(x, h.detach())
        x = x[:, -1]
        x = self.fc1(x)
        return x, h

    def init_hidden(self, batch_size):
        return zeros(1+self.__bidirectional*1, batch_size, self.__n_gru)

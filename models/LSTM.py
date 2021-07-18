import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, max_apps, seq_length, n_lstm, src_dim=4, bidirectional=True):
        super().__init__()
        self.src_embedding = nn.Embedding(num_embeddings=max_apps, embedding_dim=src_dim)
        self.lstm = nn.LSTM(input_size=src_dim, hidden_size=n_lstm, bidirectional=bidirectional, batch_first=True)
        self.fc1 = nn.Linear(in_features=n_lstm+bidirectional*n_lstm, out_features=max_apps)

        self.seq_length = seq_length

    def forward(self, x):
        x = self.src_embedding(x)
        x, _ = self.lstm(x) # Initial hidden state h0 defaults naar zeros
        x = x[:, -1] # Enkel output van laatste timestep
        x = self.fc1(x)
        return x

import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, max_apps, seq_length, n_gru, src_dim=4, bigru_dropout=0.2, bidirectional=True, zeroshot=False):
        super().__init__()
        self.src_embedding = nn.Embedding(num_embeddings=max_apps, embedding_dim=src_dim)
        self.bigru = nn.GRU(input_size=src_dim, hidden_size=n_gru, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=bigru_dropout)
        self.fc1 = nn.Linear(in_features=n_gru+bidirectional*n_gru, out_features=max_apps)

        # self.criterion = models.criterion_emb(config.hidden_size, tgt_vocab_size, use_cuda)

        self.seq_length = seq_length

    def forward(self, x):
        if len(x) != self.seq_length:
            raise ValueError("Input sequence is not of defined length: seq_length")
        x = self.src_embedding(x)
        x = x.unsqueeze(1)
        x = self.bigru(x)[0][1]
        x = self.dropout(x)
        x = self.fc1(x)
        return x

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, max_apps, src_dim=4):
        super().__init__()
        self.src_embedding = nn.Embedding(num_embeddings=max_apps, embedding_dim=src_dim)
        self.fc1 = nn.Linear(in_features=src_dim, out_features=max_apps)

    def forward(self, x):
        x = self.src_embedding(x)
        x = self.fc1(x)
        return x

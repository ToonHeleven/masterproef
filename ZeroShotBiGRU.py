import torch.nn as nn
from torch import optim
from data.dataloader import DatasetLoader
from torch.utils.data import DataLoader
import torch
import numpy as np

from tqdm import tqdm


class ZeroShotBiGRU(nn.Module):
    def __init__(self, max_apps, seq_length, n_gru, src_dim=4, bigru_dropout=0.2, bidirectional=True):
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

zeroshot = ZeroShotBiGRU(max_apps=200,
                         seq_length=10,
                         n_gru=64)
zeroshot.to(device=device)

dataset = DatasetLoader("eventlog/phone_usage_cleaned.csv", seq_length=10, max_apps=200)
removeapps = ["Screen on (locked)",
              "Screen off (locked)",
              "Screen on (unlocked)",
              "Screen off",
              "Samsung Experience Service",
              "Package installer",
              "System UI",
              "Customisation Service",
              "Configuration update",
              "EmergencyManagerService",
              "DeviceKeystring",
              "Samsung Keyboard",
              "HwModuleTest",
              "Device shutdown",
              "Device boot"]
dataset.clean(removeapps)
train_dataloader = DataLoader(dataset)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(zeroshot.parameters(), lr=0.0015)
running_loss = 0.0
i = 0
total = 0
correct = 0
epochs = 4
for prev_apps_indices, curr_app_index, curr_app_onehot in tqdm(train_dataloader):
    prev_apps_indices, curr_app_index = prev_apps_indices.to(device), curr_app_index.to(device)

    for j in range(epochs):
        optimizer.zero_grad()

        outputs = zeroshot(prev_apps_indices[0])

        loss = criterion(outputs, curr_app_index)
        loss.backward()
        optimizer.step()

        total += 1
        _, predicted = torch.topk(outputs.data, 5)
        correct += (curr_app_index in predicted)*1

        running_loss += loss.item()
    if i % 2000 == 1999:
        print('loss: %.3f' %
              (running_loss / 2000))
        running_loss = 0.0
        print('Accuracy of the network is:' + str(100 * correct / total))
        total = 0
        correct = 0
    i += 1

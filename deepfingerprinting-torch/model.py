import torch
import torch.nn as nn
from math import ceil


class DFNet(nn.Module):
    def __init__(self, input_size, classes):
        super(DFNet, self).__init__()
        filter_num = [1, 32, 64, 128, 256]
        kernel_size = [0, 8, 8, 8, 8]
        conv_stride_size = [0, 1, 1, 1, 1]
        pool_stride_size = [0, 4, 4, 4, 4]
        pool_size = [0, 8, 8, 8, 8]

        convs = []
        for i in range(1, 5):
            convs.append(nn.Sequential(
                nn.Conv1d(filter_num[i - 1], filter_num[i], kernel_size[i],
                          stride=conv_stride_size[i], padding='same'),
                nn.BatchNorm1d(filter_num[i]),
                nn.ELU() if i == 1 else nn.ReLU(),
                nn.Conv1d(filter_num[i], filter_num[i], kernel_size[i],
                          stride=conv_stride_size[i], padding='same'),
                nn.BatchNorm1d(filter_num[i]),
                nn.ELU() if i == 1 else nn.ReLU(),
                nn.MaxPool1d(pool_size[i], stride=pool_stride_size[i], ceil_mode=True),
                nn.Dropout(0.1)
            ))
            input_size = ceil((input_size - pool_size[i]) / pool_stride_size[i] + 1)
        self.convs = nn.ModuleList(convs)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filter_num[-1] * input_size, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.out = nn.Linear(512, classes)

    def forward(self, inp):
        output = inp
        for conv in self.convs:
            output = conv(output)
        output = self.fc(output)
        output = self.out(output)
        return output

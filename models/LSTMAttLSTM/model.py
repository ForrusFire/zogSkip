import torch
import torch.nn as nn

from ..attention import SeqSelfAttention


class Model(nn.Module):
    def __init__(self, LSTM1_num_units, SeqSelfAttention_num_units, LSTM2_num_units, dropout_prob, sequence_length, output_length):
        super().__init__()
        self.LSTM1 = nn.LSTM(1, LSTM1_num_units, batch_first=True, bidirectional=True)
        self.SeqSelfAttention = SeqSelfAttention(2 * LSTM1_num_units, SeqSelfAttention_num_units)
        self.LSTM2 = nn.LSTM(2 * LSTM1_num_units, LSTM2_num_units, batch_first=True)
        self.Dropout = nn.Dropout(dropout_prob)
        self.Linear = nn.Linear(LSTM2_num_units * sequence_length, output_length)

    def forward(self, x):
        x, _ = self.LSTM1(x)
        x = self.SeqSelfAttention(x)
        x = self.Dropout(x)
        x, _ = self.LSTM2(x)
        x = self.Dropout(x)
        x = self.Linear(torch.flatten(x, start_dim=1))
        return x

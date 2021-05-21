import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SeqSelfAttention(nn.Module):
    """
    Applies an attention mechanism for processing sequential data that considers the context at each timestep.

    For each element in the input sequence, the layer computes the following functions:

        h_{t, t'} = tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        e_{t, t'} = W_a h_{t, t'} + b_a
        a_{t} = softmax(e_t)
        l_t = \sum_{t'} a_{t, t'} x_{t'}

    where h_{t, t'} is the hidden state at time t with respect to t', x_t and x_{t'} are
    the input at time t and t' respectively, e_{t, t'} is the emission at time t with respect to t',
    a_{t} is the alignment at time t, and l_t is the context at time t.

    This particular layer is best paired with a bidirectional LSTM. For further details, see: https://arxiv.org/pdf/1806.01264.pdf

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
    
    Shape:
        input: (batch, seq_len, input_size) Tensor containing the features of the sequence
        output: (batch, seq_len, input_size) Tensor containing the output features l_t from the attention
            mechanism. The output has the same shape as the input.

    Attributes:
        weight_t: The learnable input-hidden weights of shape (input_size, hidden_size). The weights are initialized
            as a glorot normalized tensor.
        weight_x: The learnable input-hidden weights of the same sequence of shape (input_size, hidden_size). The weights
            are initialized as a glorot normalized tensor.
        bias_h: The learnable input-hidden bias of shape (hidden_size). The weights are initalized as zero.
        weight_a: The learnable hidden-emission weights of shape (hidden_size, 1). The weights are initalized as
            a glorot normalized tensor.
        bias_a: The learnable hidden-emission bias of shape (1). The weights are initalized as zero.

    This attention layer uses additive attention and considers the whole sequence when calculating the relevance. Note that the 
    additive attention alignment score is different than the scaled dot-product attention alignment score used in the 
    Attention is All You Need paper (Vaswani et al., 2017).
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.Wt = nn.Parameter(create_glorot_norm_tensor(input_size, hidden_size))
        self.Wx = nn.Parameter(create_glorot_norm_tensor(input_size, hidden_size))
        self.bh = nn.Parameter(create_zero_tensor(hidden_size))

        self.Wa = nn.Parameter(create_glorot_norm_tensor(hidden_size, 1))
        self.ba = nn.Parameter(create_zero_tensor(1))

    def forward(self, x):
        e = self.calculate_emission(x)

        # Apply attention activation (sigmoid) to emission
        e = torch.sigmoid(e)

        # a_{t} = \text{softmax}(e_t)
        a = F.softmax(e, dim=-1)

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        l = torch.bmm(a, x)
        return l

    def calculate_emission(self, x):
        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = torch.unsqueeze(torch.tensordot(x, self.Wt, dims=1), 2)
        k = torch.unsqueeze(torch.tensordot(x, self.Wx, dims=1), 1)
        h = torch.tanh(q + k + self.bh)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        e = torch.squeeze(torch.tensordot(h, self.Wa, dims=1) + self.ba, dim=-1)
        return e


def create_glorot_norm_tensor(fan_in, fan_out):
    # Creates a 2D glorot normal tensor with the given size
    tensor = torch.empty(fan_in, fan_out)

    stddev = math.sqrt(2 / (fan_in + fan_out))
    nn.init.normal_(tensor, mean=0.0, std=stddev)

    return tensor


def create_zero_tensor(size):
    # Creates a 1D zero tensor with the given size
    tensor = torch.empty(size)
    nn.init.zeros_(tensor)

    return tensor
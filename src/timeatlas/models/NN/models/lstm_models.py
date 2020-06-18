import torch
import torch.nn as nn


class Prediction(nn.Module):
    """
    A Long-Short Term Memory model for the prediction of the next n step in the TimeSeries
    """

    def __init__(self, n_features, n_hidden, seq_len, n_layers=1, bias=True, batch_first=False, dropout=0,
            bidirectional=False, horizon: int = 1):
        """

        An LSTM model to predict the next N time steps.

        Args:
            n_features: Number of features as input
            n_hidden: number of cells in each hidden layer
            seq_len: length of each featurs
            n_layers: number of hidden layer
            bias: If False, then the layer does not use bias weights b_ih and b_hh.
            batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
            dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout.
            bidirectional: if True, becomes a bidirectional LSTM.
            horizon: the number of returned values from the Linear output layer

        """
        super(Prediction, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.out_features = horizon

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            bias=bias,
                            batch_first=batch_first,
                            dropout=dropout,
                            bidirectional=bidirectional
                            )

        self.linear = nn.Linear(in_features=n_hidden, out_features=self.out_features)
        self.reset_hidden_state()

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden).double(),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden).double()
        )

    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(sequences.view(len(sequences), self.seq_len, -1),
                                          self.hidden)

        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred

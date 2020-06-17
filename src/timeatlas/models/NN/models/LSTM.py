import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    A Long-Short Term Memory model for the prediction of the next n step in the TimeSeries
    """

    def __init__(self, n_features, n_hidden, seq_len, n_layers=1, bias=True, batch_first=False, dropout=0,
            bidirectional=False, horizon: int = 1):
        super().__init__()
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

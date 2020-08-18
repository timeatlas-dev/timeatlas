from typing import List

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from timeatlas.abstract import AbstractBaseModel

class LSTMPrediction(nn.Module, AbstractBaseModel):

    def __init__(self, n_features, n_hidden, seq_len, optimizer: Optimizer, loss_function, n_layers=1,
            bias=True, batch_first=False, dropout=0, bidirectional=False, horizon: int = 1, **kwargs):
        """

        Args:
            model: object instance of a LSTM from timeatlas (e.g. model=Prediction(<parameters>)
            epochs: number of epochs on the training set
            optimizer: class of the optimizer -> see Pytroch -> torch.optim
            loss_function: a function that calculates the loss -> see PyTorch -> torch.nn
            n_features: Number of features as input
            n_hidden: number of cells in each hidden layer
            seq_len: length of each featurs
            n_layers: number of hidden layer
            bias: If False, then the layer does not use bias weights b_ih and b_hh.
            batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
            dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout.
            bidirectional: if True, becomes a bidirectional LSTM.
            horizon: the number of returned values from the Linear output layer
            **kwargs: kwargs for the optimizer
        """

        super(LSTMPrediction, self).__init__()

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

        # optimizer and loss function for the fit
        self.optimizer = optimizer(self.parameters(), **kwargs)
        self.loss_function = loss_function

        # saving the history
        self.loss = []

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

    def fit(self, series: DataLoader, epochs: int = 10, verbose: bool = True):
        super().fit(series)
        for i in range(epochs):
            for seq, labels in self.X_train:

                self.reset_hidden_state()

                self.optimizer.zero_grad()

                y_pred = self(seq)

                single_loss = self.loss_function(y_pred.squeeze(), labels)
                single_loss.backward()
                self.optimizer.step()

                self.loss.append(single_loss.item())

            if verbose:
                if i in range(0, epochs + round(epochs / 10), round(epochs / 10)):
                    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        if verbose:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    def predict(self, horizon: int = 1, testset: DataLoader = None) -> List:
        super().predict(horizon)

        if testset is None:
            raise Exception(f'the testset has to be of type {type(DataLoader)}, got {type(testset)}')

        self.eval()
        predictions = []
        for seq, l in testset:
            y_pred = self(seq)
            predictions.append(y_pred)

        return predictions

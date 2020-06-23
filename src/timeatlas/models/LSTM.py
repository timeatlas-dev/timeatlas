from typing import List

from timeatlas.abstract import AbstractBaseModel

from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class LSTM(AbstractBaseModel):

    def __init__(self, model: nn.Module, epochs: int, optimizer: Optimizer, loss_function, **kwargs):
        """

        Args:
            model: object instance of a LSTM from timeatlas (e.g. model=Prediction(<parameters>)
            epochs: number of epochs on the training set
            optimizer: class of the optimizer -> see Pytroch -> torch.optim
            loss_function: a function that calculates the loss -> see PyTorch -> torch.nn
            **kwargs: kwargs for the optimizer
        """
        super(LSTM, self).__init__()

        # parameters of the LSTM
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer(self.model.parameters(), **kwargs)
        self.loss_function = loss_function

        # saving the history
        self.loss = []

    def fit(self, series: DataLoader, verbose: bool = True):
        super().fit(series)
        for i in range(self.epochs):
            for seq, labels in series:

                self.model.reset_hidden_state()

                self.optimizer.zero_grad()

                y_pred = self.model(seq)

                single_loss = self.loss_function(y_pred.squeeze(), labels)
                single_loss.backward()
                self.optimizer.step()

                self.loss.append(single_loss.item())

            if verbose:
                if i in range(0, self.epochs + round(self.epochs / 10), round(self.epochs / 10)):
                    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        if verbose:
            print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    def predict(self, horizon: int = 1, testset: DataLoader = None) -> List:
        super().predict(horizon)

        if testset is None:
            raise Exception(f'the testset has to be of type {type(DataLoader)}, got {type(testset)}')

        self.model.eval()
        predictions = []
        for seq, l in testset:
            y_pred = self.model(seq)
            predictions.append(y_pred)

        return predictions

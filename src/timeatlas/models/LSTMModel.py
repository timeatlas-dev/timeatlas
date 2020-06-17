from typing import NoReturn

from timeatlas.abstract import AbstractBaseModel

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class LSTMModel(AbstractBaseModel):

    def __init__(self, model: nn.Module, epochs: int, optimizer: Optimizer, loss_function, *args, **kwargs):
        super(LSTMModel, self).__init__()
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer(self.model.parameters(), **kwargs)
        self.loss_function = loss_function

    def fit(self, series: DataLoader, verbose: bool = True):
        super().fit(series)
        for i in range(self.epochs):
            for seq, labels in series:

                self.model.reset_hidden_state()

                self.optimizer.zero_grad()

                y_pred = self.model(seq)

                single_loss = self.loss_function(y_pred, labels)
                single_loss.backward()
                self.optimizer.step()

            if verbose:
                if i % 25 == 1:
                    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        if verbose:
            print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    def predict(self, horizon: int, testset: DataLoader = None) -> NoReturn:
        super().predict(horizon)

        if testset is None:
            raise Exception(f'testset has to be of type {type(DataLoader)}, got {type(testset)}')

        self.model.eval()
        predictions = []
        for seq in testset:
            with torch.no_grad():
                self.model.reset_hidden_state()
                predictions.append(self.model(seq).item())

        return predictions

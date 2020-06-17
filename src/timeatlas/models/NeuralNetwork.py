from timeatlas.abstract import AbstractBaseModel

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class NeuralNetwork(AbstractBaseModel):

    def __init__(self, model: nn.Module, epochs: int, optimizer: Optimizer, loss_funciton):
        super(NeuralNetwork, self).__init__()
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer(self.model.parameters(), lr=0.01)
        self.loss_function = loss_funciton

    def fit(self, series: DataLoader):
        for i in range(self.epochs):
            for seq, labels in series:

                self.model.reset_hidden_state()

                self.optimizer.zero_grad()

                y_pred = self.model(seq)

                single_loss = self.loss_function(y_pred, labels)
                single_loss.backward()
                self.optimizer.step()

            if i % 25 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    def predict(self, horizon):
        pass

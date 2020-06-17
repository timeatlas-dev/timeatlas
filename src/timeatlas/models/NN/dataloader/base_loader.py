from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler


class BaseDataset(Dataset):

    def __init__(self, data=None):
        self.data = data

    def min_max_norm(self):
        scaler = MinMaxScaler()
        return scaler.fit_transform(self.data.T).T

    def z_score_norm(self):
        raise NotImplementedError

    def sigmoid_norm(self):
        raise NotImplementedError

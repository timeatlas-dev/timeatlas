from pandas import Series


class Scaler:

    @staticmethod
    def minmax(s: Series) -> Series:
        scaled_series = (s - s.min()) / (s.max() - s.min())
        return scaled_series

    @staticmethod
    def zscore(s: Series) -> Series:
        scaled_series = (s - s.mean()) / s.std()
        return scaled_series

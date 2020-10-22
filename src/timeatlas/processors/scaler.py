from pandas import Series


class Scaler:

    @staticmethod
    def minmax(s: Series) -> Series:
        """ minmax scaling

        Normalize the time-series based on min and max in the time-series

        Args:
            s: time-series to normalize

        Returns: Series

        """
        scaled_series = (s - s.min()) / (s.max() - s.min())
        return scaled_series

    @staticmethod
    def zscore(s: Series) -> Series:
        """znorm scaling

        Normalize the time-series based on zscore

        Args:
            s: time-series to normalize

        Returns: Series

        """
        scaled_series = (s - s.mean()) / s.std()
        return scaled_series

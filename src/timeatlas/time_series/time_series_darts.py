from typing import NoReturn, Tuple, Any, Union, Optional, List

from darts import TimeSeries
import pandas as pd

from timeatlas.manipulator import TimeShop

import matplotlib.pyplot as plt


class TimeSeriesDarts(TimeSeries):

    def __init__(self, xa):
        super().__init__(xa=xa)

    @staticmethod
    def from_dataframe(df: pd.DataFrame,
            time_col: Optional[str] = None,
            value_cols: Optional[Union[List[str], str]] = None,
            fill_missing_dates: Optional[bool] = False,
            freq: Optional[str] = None, ):
        ts = super(TimeSeriesDarts, TimeSeriesDarts).from_dataframe(df=df)
        return TimeSeriesDarts(xa=ts._xa)


    def plot(self,
             new_plot: bool = False,
             central_quantile: Union[float, str] = 0.5,
             low_quantile: Optional[float] = 0.05,
             high_quantile: Optional[float] = 0.95,
             *args,
             **kwargs):
        super(TimeSeriesDarts, self).plot()
        plt.show()

    def edit(self):
        return TimeShop(self)

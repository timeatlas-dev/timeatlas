from darts import TimeSeries
from timeatlas.manipulator import TimeShop


class TimeSeriesDarts(TimeSeries):

    def __init__(self, df):
        super().__init__(df=df)

    def edit(self):
        return TimeShop(self)

from u8timeseries import TimeSeries


class Utils:

    def to_u8(self):
        return TimeSeries.from_times_and_values(self.series.index, self.series.values)

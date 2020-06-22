from timeatlas import TimeSeries


def minmax(ts: TimeSeries) -> TimeSeries:
    s = ts.series
    scaled_series = (s - s.min()) / (s.max() - s.min())
    return TimeSeries(scaled_series, ts.metadata)


def zscore(ts: TimeSeries) -> TimeSeries:
    s = ts.series
    scaled_series = (s - s.mean()) / s.std()
    return TimeSeries(scaled_series, ts.metadata)

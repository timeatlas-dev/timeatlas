import time
import pandas as pd
from timeatlas import TimeSeries, TimeSeriesDataset

from timeatlas import AnomalyGenerator

# TODO: This test is pretty bad...I need to rework it (Lorenz)

def test_anomaly_generator():
    from datetime import datetime

    start_time = time.time()
    print("---START DATASET CREATION---")
    # data_reader = Reader('../../data/clean/ACS-F2/Dataset/*/*.xml')
    data = pd.read_csv('./anomaly_generator_test_data.csv', index_col=0).T
    data.drop('label', inplace=True)
    tss = [datetime.fromtimestamp(int(x)) for x in data.index]
    result = []

    for i, v in data.iteritems():
        v.index = tss[:len(v)]
        result.append(TimeSeries(v))

    tsd = TimeSeriesDataset(result)

    anomaly_set = AnomalyGenerator(data=tsd, conf_file='test_config.ini')
    anomaly_set.generate()
    end_time = time.time()
    print("Creation time of the dataset: {0:.3f} seconds.".format(end_time - start_time))


if __name__ == '__main__':
    test_anomaly_generator()

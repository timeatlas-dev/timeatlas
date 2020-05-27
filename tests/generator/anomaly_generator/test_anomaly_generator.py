import time
import pandas as pd

from timeatlas import AnomalyGenerator

# TODO: This test is pretty bad...I need to rework it (Lorenz)

def test_anomaly_generator():
    start_time = time.time()
    print("---START DATASET CREATION---")
    # data_reader = Reader('../../data/clean/ACS-F2/Dataset/*/*.xml')
    data = pd.read_csv('./anomaly_generator_test_data.csv', index_col=0)
    anomaly_set = AnomalyGenerator(data=data, conf_file='config.ini')
    anomaly_set.generate()
    end_time = time.time()
    print("Creation time of the dataset: {0:.3f} seconds.".format(end_time - start_time))


if __name__ == '__main__':
    test_anomaly_generator()

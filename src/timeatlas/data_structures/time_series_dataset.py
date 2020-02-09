from dataclasses import dataclass
from typing import Dict

from .time_series import TimeSeries, Sensor


@dataclass
class TimeSeriesDataset:
    """

    Un TimeSeriesDataset est une collection de series temporelles diverses.
    - ajout, etc. comme dans une list en modifiant l'objet data
    - modifications? Comment faire?
        - Faut il que un TSD soit immuable ou non?
    
    """

    data: list[TimeSeries]


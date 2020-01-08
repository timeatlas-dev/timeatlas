from dataclasses import dataclass
from typing import Dict
import pandas as pd

@dataclass
class Dataset:
    objects: pd.DataFrame
    values: Dict[int, pd.Series]

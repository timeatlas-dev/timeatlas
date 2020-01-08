from dataclasses import dataclass
from .unit import Unit

@dataclass
class Object:
    id: int
    name: str
    unit: Unit


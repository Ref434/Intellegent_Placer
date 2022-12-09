from dataclasses import dataclass
from imageio.core.util import Array
from intelligent_placer_lib.config import Cases

@dataclass
class Image():

    name: str = None
    case: Cases = None  # values : True, False, Error
    origin: Array = None
    mask = None
    polygon = None
    result: Cases = None  # values : True, False, Error
    polygon_area: float = None
    items_area: float = None
    items = []
    result_image = None
    test = None
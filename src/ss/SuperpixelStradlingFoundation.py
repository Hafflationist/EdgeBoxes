from dataclasses import dataclass

from typing import List, Set, Tuple


@dataclass
class SuperpixelStradlingFoundation:
    segmentation: List[Set[Tuple[int, int]]]
    scale: float

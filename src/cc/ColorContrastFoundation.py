from dataclasses import dataclass

from numpy.core.multiarray import ndarray


@dataclass
class ColorContrastFoundation:
    img_lab: ndarray

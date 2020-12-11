from dataclasses import dataclass

from numpy.core.multiarray import ndarray


@dataclass
class MultiscaleSaliencyFoundation:
    saliency: ndarray

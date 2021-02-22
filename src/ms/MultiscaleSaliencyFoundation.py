from dataclasses import dataclass

from numpy.core.multiarray import ndarray


@dataclass
class MultiscaleSaliencyFoundation:
    saliency_1: ndarray
    saliency_2: ndarray
    saliency_3: ndarray
    saliency_4: ndarray
    saliency_5: ndarray

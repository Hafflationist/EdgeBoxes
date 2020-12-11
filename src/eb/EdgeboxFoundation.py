from dataclasses import dataclass

from numpy.core.multiarray import ndarray


@dataclass
class EdgeboxFoundation:
    edges_nms: ndarray
    edges_with_grouping: ndarray
    groups_members: ndarray
    affinities: ndarray

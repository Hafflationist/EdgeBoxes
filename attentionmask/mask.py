import numpy as np
import math

from pycocotools import mask as _mask

from attentionmask.type_assert import *

EPS = 1e-9

#  bbs     - [nx4] Bounding box(es) stored as [x y w h]

# fix ndarray order problem
# you don't need to transpose((1, 2, 0))

def _masks_as_fortran_order(masks):
    masks = masks.transpose((1, 2, 0))
    masks = np.asfortranarray(masks)
    masks = masks.astype(np.uint8)
    return masks


def _masks_as_c_order(masks):
    masks = masks.transpose((2, 0, 1))
    masks = np.ascontiguousarray(masks)
    return masks


def encode(obj):
    # return single RLE
    if len(obj.shape) == 2:
        mask = obj
        masks = np.array(np.asarray([mask]))
        masks = _masks_as_fortran_order(masks)
        rles = _mask.encode(masks)
        rle = rles[0]
        return rle
    # return RLEs
    elif len(obj.shape) == 3:
        masks = obj
        masks = _masks_as_fortran_order(masks)
        rles = _mask.encode(masks)
        return rles
    else:
        raise Exception("Not Implement")


def decode(obj):
    # return single mask
    if is_RLE(obj):
        rles = [obj]
        masks = _mask.decode(rles)
        masks = _masks_as_c_order(masks)
        mask = masks[0]
        return mask
    # return masks
    elif is_RLEs(obj):
        rles = obj
        masks = _mask.decode(rles)
        masks = _masks_as_c_order(masks)
        return masks
    else:
        raise Exception("Not Implement")

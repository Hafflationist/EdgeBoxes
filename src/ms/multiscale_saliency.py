from typing import Tuple

import numpy as np
import numpy.fft
from numpy.core.multiarray import ndarray
from skimage.transform import resize
from skimage.filters import gaussian
from scipy.ndimage import convolve

from src.ms.MultiscaleSaliencyFoundation import MultiscaleSaliencyFoundation


def __calculate_one_channel(img_mono: ndarray) -> ndarray:
    complex_img = numpy.fft.fft2(img_mono)
    img_phase: ndarray = np.angle(complex_img)
    img_freq: ndarray = np.abs(complex_img)
    img_freq = np.log(img_freq)
    img_freq_smooth = convolve(img_freq, (np.ones((3, 3)) / 9.0), mode='reflect')
    img_freq_residual = (img_freq - img_freq_smooth)
    # TODO Falls noch Zeit ist, muss man hier schauen, ob man wegen des Paddings nicht vllt fftshift anwenden sollte
    result = np.abs(np.real(numpy.fft.ifft2(np.exp(img_freq_residual) * np.exp(img_phase * 1.0j)))) ** 2.0
    # Folgendes muss passieren, um Kantensalienz zu beseitigen
    result[0, :] = 0
    result[:, 0] = 0
    result[-1, :] = 0
    result[:, -1] = 0
    result_smooth = gaussian(result, sigma=1.0)
    result_smooth /= np.max(result_smooth)
    return result_smooth


def __calculate_multiscale_saliency(img_orig: ndarray, scale: int) -> ndarray:
    # Selbst wenn man keine Größenveränderung braucht, muss hier resize stehen
    # Grund: unbekannt
    # Konsequenz beim Entfernen: IFFT funktioniert nicht wie erwartet
    img: ndarray = resize(img_orig, (16 * scale, 16 * scale))
    img_red: ndarray = np.array([[px[0] for px in row] for row in img])
    img_green: ndarray = np.array([[px[1] for px in row] for row in img])
    img_blue: ndarray = np.array([[px[2] for px in row] for row in img])
    saliency_red: ndarray = __calculate_one_channel(img_red)
    saliency_green: ndarray = __calculate_one_channel(img_green)
    saliency_blue: ndarray = __calculate_one_channel(img_blue)
    saliency: ndarray = (saliency_red + saliency_green + saliency_blue) / 3
    return resize(saliency, (img_orig.shape[0], img_orig.shape[1]))


def image_2_foundation(img: ndarray) -> MultiscaleSaliencyFoundation:
    return MultiscaleSaliencyFoundation(__calculate_multiscale_saliency(img, 1),
                                        __calculate_multiscale_saliency(img, 2),
                                        __calculate_multiscale_saliency(img, 3))


def get_objectness(foundation: MultiscaleSaliencyFoundation,
                   mask_coords: ndarray,
                   theta_ms: float = 0.0,
                   learned: bool = False) -> Tuple[float, float, float]:
    if not learned:
        theta_ms = 2.0 / 3.0

    def scale_specific(saliency: ndarray) -> float:
        mask_values = np.array(list(map(lambda idx: saliency[idx[0], idx[1]], mask_coords)))
        max_post = np.max(saliency) * theta_ms
        mask_values_filtered = list(filter(lambda p: p >= max_post, mask_values))
        mask_n = len(mask_coords)
        return np.sum(mask_values_filtered) * float(len(mask_values_filtered) / float(mask_n))

    scale_1_obj = scale_specific(foundation.saliency_1)
    scale_2_obj = scale_specific(foundation.saliency_2)
    scale_3_obj = scale_specific(foundation.saliency_3)
    return scale_1_obj, scale_2_obj, scale_3_obj

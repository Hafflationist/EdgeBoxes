import numpy as np
import numpy.fft
from numpy.core.multiarray import ndarray
from skimage.transform import resize
from skimage.filters import gaussian
from scipy.ndimage import convolve


def __calculate_one_channel(img_mono: ndarray) -> ndarray:
    complex_img = numpy.fft.fft2(img_mono)
    img_phase = np.angle(complex_img)
    img_freq = np.abs(complex_img)
    img_freq = np.log(img_freq)
    img_freq_smooth = convolve(img_freq, (np.ones((3, 3)) / 9.0), mode='reflect')
    img_freq_residual = (img_freq - img_freq_smooth)
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
    img = resize(img_orig, (16 * scale, 16 * scale))
    img_red = np.array([[px[0] for px in row] for row in img])
    img_green = np.array([[px[1] for px in row] for row in img])
    img_blue = np.array([[px[2] for px in row] for row in img])
    saliency_red = __calculate_one_channel(img_red)
    saliency_green = __calculate_one_channel(img_green)
    saliency_blue = __calculate_one_channel(img_blue)
    saliency = (saliency_red + saliency_green + saliency_blue) / 3
    return saliency


def get_objectness(img: ndarray, mask: ndarray, theta_ms: float = 0.0, learned: bool = False) -> float:
    saliency = __calculate_multiscale_saliency(img, 1)  # calculate integral image to boost computation across multiple proposals
    if not learned:
        theta_ms = np.max(saliency) * (2.0 / 3.0)
    mask_coords = np.transpose(np.where(mask))
    mask_values = list(map(lambda idx: saliency[idx], mask_coords))
    mask_values_filtered = filter(lambda p: p >= theta_ms, mask_values)
    mask_n = len(mask_coords)
    return np.sum(mask_values_filtered) * float(len(mask_values) / float(mask_n))

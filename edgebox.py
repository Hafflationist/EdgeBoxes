import cv2
import numpy as np
import colorsys


def detect_edges(img):
    img_processed = (img / np.max(img)).astype(np.float32)
    modelFilename = "model/model.yml.gz"
    pDollar = cv2.ximgproc.createStructuredEdgeDetection(modelFilename)
    edges = pDollar.detectEdges(cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))
    orientation_map = pDollar.computeOrientation(edges)
    edges_nms = pDollar.edgesNms(edges, orientation_map)
    edges_nms[edges_nms < 0.1] = 0      # thresholding

    def calculate_pixel(row_idx, px_idx):
        if edges_nms[row_idx, px_idx] < 0.1:
            return [0.0, 0.0, 0.0, 0.0]
        o = orientation_map[row_idx, px_idx]
        pi = 3.14159265358979
        rgb = colorsys.hsv_to_rgb(o/pi, 1.0, 1.0)
        return [1.0, rgb[0], rgb[1], rgb[2]]

    edges_nms_colored = [[calculate_pixel(row_idx, px_idx) for px_idx in range(len(edges_nms[0]))] for row_idx in range(len(edges_nms))]

    # todo iterating to get edge groups
    # for row_idx in range(len(edges_nms)):
    #     for px_idx in range(len(edges_nms)):
    #         if edges_nms[row_idx, px_idx] > 0.1:
    #             edges_nms[row_idx, px_idx] = orientation_map[row_idx, px_idx]
    #             print("Pixel " + str(edges_nms[row_idx, px_idx]) + " with " + str(orientation_map[row_idx, px_idx]))

    edges_nms = edges_nms / np.max(edges_nms)
    return np.array(edges_nms_colored), orientation_map

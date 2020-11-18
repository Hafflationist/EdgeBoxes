import cv2

import edgebox as eb

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_img = cv2.imread("assets/testImage.jpg")
    cv2.imshow("hugo", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    test_edges_nms, orientation_map = eb.detect_edges(test_img)
    cv2.imshow("nms", test_edges_nms)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

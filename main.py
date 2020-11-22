import cv2

import edgebox as eb

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_img = cv2.imread("assets/testImage.jpg")
    # cv2.imshow("hugo", test_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    test_edges_nms, orientation_map = eb.detect_edges(test_img)
    # cv2.imshow("nms", test_edges_nms * 255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    test_edges_nms_colored = eb.color_edges(test_edges_nms, orientation_map)
    cv2.imshow("nms (colored)", test_edges_nms_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    hugo = eb.color_grouped_edges(eb.group_edges(test_edges_nms, orientation_map))
    cv2.imshow("nms grouped (colored)", hugo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("finished#")

import cv2
import datetime

import edgebox as eb


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_img = cv2.imread("assets/testImage.jpg")
    test_img = cv2.imread("assets/testImage2.jpg")

    test_edges_nms, orientation_map = eb.detect_edges(test_img)
    # cv2.imshow("nms", test_edges_nms * 255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    test_edges_nms_colored = eb.color_edges(test_edges_nms, orientation_map)
    # cv2.imshow("nms (colored)", test_edges_nms_colored)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    b = datetime.datetime.now()
    test_edges_nms_grouped, groups_members = eb.group_edges(test_edges_nms, orientation_map)
    a = datetime.datetime.now()
    print("group_edges:\t" + str(a - b))

    b = datetime.datetime.now()
    test_edges_nms_grouped_colored = eb.color_grouped_edges(test_edges_nms_grouped)
    a = datetime.datetime.now()
    print("color_grouped_edges:\t" + str(a - b))

    cv2.imshow("nms grouped (colored)", test_edges_nms_grouped_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("finished#")

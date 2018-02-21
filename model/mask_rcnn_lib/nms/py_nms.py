import numpy as np

# deafult nms in python for checking
def py_nms(dets, thresh):
    x0 = dets[:, 0]
    y0 = dets[:, 1]
    x1 = dets[:, 2]
    y1 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x1 - x0 + 1) * (y1 - y0 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx0 = np.maximum(x0[i], x0[order[1:]])
        yy0 = np.maximum(y0[i], y0[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])

        w = np.maximum(0.0, xx1 - xx0 + 1)
        h = np.maximum(0.0, yy1 - yy0 + 1)
        intersect = w * h
        overlap = intersect / (areas[i] + areas[order[1:]] - intersect)

        inds  = np.where(overlap <= thresh)[0]
        order = order[inds + 1]

    return keep
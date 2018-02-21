from common import *
from utility.draw import *


from model.mask_rcnn_lib.box import *
from dataset.reader import *


#different draw functions for debugging  ------
def to_color(s, color):
    color = s*np.array(color).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def unflat(data,  W, H, C=None ):
    datas=[]
    i=0
    for h,w in zip(H,W):
        if C is None:
            d = data[i:i+h*w].reshape(h,w)
            i += h*w
        else:
            d = data[i:i+C*h*w].reshape(h,w,C)
            i += C*h*w
        datas.append(d)
    return datas

## ground truth **

def draw_gt_boxes(image, gt_boxes):
    image = image.copy()
    gt_boxes= gt_boxes.astype(int)
    for box in gt_boxes:
        x0,y0,x1,y1 =box
        #cv2.rectangle(image, (x0,y0), (x1,y1), (0,255,255), 1)
        draw_screen_rect(image, (x0,y0), (x1,y1), (0,255,255), 0.2)

    return image



def draw_label_as_gt_boxes(image, label):
    image = image.copy()

    gt_boxes = label_to_box(label)
    if gt_boxes is not None:
        return draw_gt_boxes(image, gt_boxes)

    return image






## rpn ********************************

def draw_rpn_proposal_before_nms(image, prob_flat, delta_flat, windows, threshold=0.95):
    #prob  = prob_flat.cpu().data.numpy()
    #delta = delta_flat.cpu().data.numpy()

    image = image.copy()
    height,width = image.shape[0:2]
    prob  = prob_flat
    delta = delta_flat
    index = np.argsort(prob)  #sort descend #[::-1]

    if threshold<0:
        threshold = np.percentile(prob,99.8)

    num_windows = len(windows)
    for i in index:
        #if insides[i]==0: continue #ignore bounday

        s = prob[i]
        if s<threshold:  continue

        w = windows[i]
        d = delta[i]
        b = box_transform_inv(w.reshape(1,4), d.reshape(1,4))
        b = clip_boxes(b, width, height)
        b = b.reshape(-1)


        color_w = to_color(s, [255,255,255])
        color_b = to_color(s, [0,0,255])
        #draw_dotted_rect(image,(w[0], w[1]), (w[2], w[3]),color_w , 1)
        cv2.rectangle(image,(b[0], b[1]), (b[2], b[3]), color_b, 1)
    #print((prob>threshold).sum())
    return image


def draw_rpn_proposal_after_nms(image, proposals, top=100):
    #proposals = proposals.cpu().data.numpy().reshape(-1,5)

    image = image.copy()
    boxes  = proposals[:,1:5].astype(np.int32)
    scores = proposals[:,5]
    num_proposals = len(proposals)

    index = np.argsort(scores)     ##sort descend #[::-1]
    if num_proposals>top:
        index = index[-top:]

    num =len(index)
    for n in range(num):
        i = index[n]
        s = scores[i]
        box = boxes[i]
        if s<0: continue

        #if s>0.95:
        if 1:
            color = to_color(n/num, [255,255,255])
            cv2.rectangle(image,(box[0], box[1]), (box[2], box[3]), color, 1)

    # gt_proposal for training
    # index = np.where(scores<0)[0]
    # for i in index:
    #     s  = scores[i]
    #     box = boxes[i]
    #     cv2.rectangle(image,(box[0], box[1]), (box[2], box[3]), (0,255,0), 1)
    #print(num)
    return image

# def draw_rpn_predict(image, prob_flat, delta_flat, feature_widths, feature_heights, num_bases, threshold=0.95):
#     #label        = label.cpu().data.numpy()
#     #label_weight = label_weight.cpu().data.numpy()
#
#     label = label*128 + 127
#     label_weight = label_weight*255
#
#     label = label.astype(np.uint8)
#     label_weight = label_weight.astype(np.uint8)
#
#     labels        = unflat(label,        feature_widths, feature_heights, num_bases)
#     label_weights = unflat(label_weight, feature_widths, feature_heights, num_bases)
#
#     H,W = feature_widths[0], feature_heights[0]
#     num_heads = len(feature_heights)
#     for h in range(num_heads):
#         labels[h] = cv2.resize(labels[h], (W,H), interpolation=cv2.INTER_LINEAR)
#         label_weights[h] = cv2.resize(label_weights[h], (W,H), interpolation=cv2.INTER_LINEAR)
#
#     labels = np.hstack(labels)
#     label_weights = np.hstack(label_weights)
#
#
#     return labels, label_weights
#
#     return image






## rcnn ********************************
def draw_rcnn_detection_nms(image, detections, threshold=0.8):

    image = image.copy()
    for det in detections:
        s = det[5]
        if s<threshold: continue

        b = det[1:5]
        color = to_color(s, [255,0,255])
        cv2.rectangle(image,(b[0], b[1]), (b[2], b[3]), color, 1)

            # name  = dataset.annotation.NAMES[j]
            # text  = '%02d %s : %0.3f'%(label,name,s)
            # fontFace  = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale = 0.5
            # textSize = cv2.getTextSize(text, fontFace, fontScale, 2)
            # cv2.putText(img, text,(b[0], (int)((b[1] + 2*textSize[1]))), fontFace, fontScale, (0,0,0), 2, cv2.LINE_AA)
            # cv2.putText(img, text,(b[0], (int)((b[1] + 2*textSize[1]))), fontFace, fontScale, (255,255,255), 1, cv2.LINE_AA)
    return image

#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #check_layer()

 
 
from common import *

from model.mask_rcnn_lib.box import *
from model.mask_rcnn_lib.draw import *

#
#
# def draw_rcnn_pre_nms(image, probs, deltas, proposals, cfg, colors, names, threshold=-1, is_before=1, is_after=1):
#
#     height,width = image.shape[0:2]
#     num_classes  = cfg.num_classes
#
#     probs  = probs.cpu().data.numpy()
#     deltas = deltas.cpu().data.numpy()
#     proposals    = proposals.data.cpu().numpy()
#     num_proposals = len(proposals)
#
#     labels = np.argmax(probs,axis=1)
#     probs  = probs[range(0,num_proposals),labels]
#     idx    = np.argsort(probs)
#     for j in range(num_proposals):
#         i = idx[j]
#
#         s = probs[i]
#         l = labels[i]
#         if s<threshold or l==0:
#             continue
#
#         a = proposals[i, 0:4]
#         t = deltas[i,l*4:(l+1)*4]
#         b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
#         b = clip_boxes(b, width, height)  ## clip here if you have drawing error
#         b = b.reshape(-1)
#
#         #a   = a.astype(np.int32)
#         color = (s*np.array(colors[l])).astype(np.uint8)
#         color = (int(color[0]),int(color[1]),int(color[2]))
#         if is_before==1:
#             draw_dotted_rect(image,(a[0], a[1]), (a[2], a[3]), color, 1)
#             #cv2.rectangle(image,(a[0], a[1]), (a[2], a[3]), (int(color[0]),int(color[1]),int(color[2])), 1)
#
#         if is_after==1:
#             cv2.rectangle(image,(b[0], b[1]), (b[2], b[3]), color, 1)
#
#         draw_shadow_text(image , '%f'%s,(b[0], b[1]), 0.5, (255,255,255), 1, cv2.LINE_AA)
#
#
#
# #---------------------------------------------------------------------------
#
#this is in cpu: <todo> change to gpu ?
def rcnn_nms(cfg, mode, inputs, proposals, probs, deltas ):

    if mode in ['train',]:
        nms_pre_threshold          = cfg.rcnn_train_nms_pre_threshold
        nms_post_overlap_threshold = cfg.rcnn_train_nms_post_overlap_threshold
        nms_max_per_image          = cfg.rcnn_train_nms_max_per_image

    elif mode in ['valid', 'test',]:
        nms_pre_threshold          = cfg.rcnn_test_nms_pre_threshold
        nms_post_overlap_threshold = cfg.rcnn_test_nms_post_overlap_threshold
        nms_max_per_image          = cfg.rcnn_test_nms_max_per_image

    elif mode in ['eval']:
        nms_pre_threshold = 0.05 ##0.05   # set low numbe r to make roc curve.
                                          # else set high number for faster speed at inference
        nms_post_overlap_threshold = cfg.rcnn_test_nms_post_overlap_threshold
        nms_max_per_image          = cfg.rcnn_test_nms_max_per_image

    else:
        raise ValueError('rcnn_nms(): invalid mode = %s?'%mode)


    batch_size = len(inputs)
    height, width = (inputs.size(2),inputs.size(3))  #original image width
    num_classes = cfg.num_classes

    probs  = probs.cpu().data.numpy()
    deltas = deltas.cpu().data.numpy().reshape(-1, num_classes,4)
    proposals = proposals.cpu().data.numpy()

    #non-max suppression
    detections = []
    for b in range(batch_size):
        detection = [np.empty((0,7),np.float32),]

        idx = np.where(proposals[:,0]==b)[0]
        if len(idx)>0:
            ps = probs [idx]
            ds = deltas[idx]
            proposal = proposals[idx]

            for j in range(1,num_classes): #skip background
                idx = np.where(ps[:,j] > nms_pre_threshold)[0]
                if len(idx)>0:
                    p = ps[idx, j].reshape(-1,1)
                    d = ds[idx, j]
                    box = box_transform_inv(proposal[idx,1:5], d)
                    box = clip_boxes(box, width, height)
                    keep = gpu_nms(np.hstack((box, p)), nms_post_overlap_threshold)

                    num = len(keep)
                    det = np.zeros((num,7),np.float32)
                    det[:,0  ] = b
                    det[:,1:5] = box[keep]
                    det[:,5  ] = p[keep,0]
                    det[:,6  ] = j
                    detection.append(det)

        detection = np.vstack(detection)

        ##limit to MAX_PER_IMAGE detections over all classes
        if nms_max_per_image > 0:
            if len(detection) > nms_max_per_image:
                threshold = np.sort(detection[:,4])[-nms_max_per_image]
                keep = np.where(detection[:,4] >= threshold)[0]
                detection = detection[keep, :]

        detections.append(detection)

    return detections




#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



 
 
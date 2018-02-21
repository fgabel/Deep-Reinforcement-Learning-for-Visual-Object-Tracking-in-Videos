from common import *
from model.mask_rcnn_lib.box import *
from model.mask_rcnn_lib.draw import *

#from net.lib.mxnet_mask_rcnn.box import *
#from dataset.draw import *
#
#


def make_fake_masks(cfg, mode, inputs, detections, ):#<todo>
    masks = []
    batch_size,C,H,W = inputs.size()
    for detection in detections:
        mask = np.zeros((H, W), np.float32)
        masks.append(mask)
    return masks



def mask_nms( cfg, mode, inputs, proposals, mask_probs):

    nms_threshold = cfg.mask_test_nms_threshold
    threshold = cfg.mask_test_threshold


    mask_probs  = mask_probs.cpu().data.numpy()
    proposals = proposals.cpu().data.numpy()

    masks = []
    batch_size,C,H,W = inputs.size()
    for b in range(batch_size):
        mask  = np.zeros((H,W),np.float32)
        index = np.where(proposals[:,0]==b)[0]

        instance_id=1
        if len(index) != 0:
            for i in index:
                p = proposals[i]
                prob = p[5]
                #print(prob)
                if prob>nms_threshold:
                    x0,y0,x1,y1 = p[1:5].astype(np.int32)
                    h, w = y1-y0+1, x1-x0+1
                    label = int(p[6]) #<todo>
                    crop = mask_probs[i, label]
                    crop = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)
                    crop = crop>threshold

                    mask[y0:y1+1,x0:x1+1] = crop*instance_id + (1-crop)*mask[y0:y1+1,x0:x1+1]
                    instance_id = instance_id+1

                if 0: #<debug>

                    images = inputs.data.cpu().numpy()
                    image = (images[b].transpose((1,2,0))*255).astype(np.uint8)
                    image = np.clip(image.astype(np.float32)*4,0,255)

                    image_show('image',image,2)
                    image_show('mask',mask/mask.max()*255,2)
                    cv2.waitKey(0)

            #<todo>
            #non-max-suppression to remove overlapping segmentation

        masks.append(mask)
    return masks


##-----------------------------------------------------------------------------  
#if __name__ == '__main__':
#    print( '%s: calling main function ... ' % os.path.basename(__file__))
#
#
#
# 
 

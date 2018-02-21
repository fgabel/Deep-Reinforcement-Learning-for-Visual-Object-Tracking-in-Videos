from common import *
from model.mask_rcnn_lib.box import *
from model.mask_rcnn_lib.draw import *


def detections_to_proposals(detections):

    detections = np.vstack(detections)
    if detections.size==0:  #<todo>
        dummy = Variable(torch.zeros(1, 7)).cuda()
        dummy[0,0] = -1
        return dummy

    proposals = Variable(torch.from_numpy(detections)).cuda()
    return proposals

def add_truth_boxes_to_proposals(cfg, proposals, truth_boxes):
    batch_size = len(truth_boxes)
    sampled_proposals = []
    for b in range(batch_size):
        combine = []
        truth_box = truth_boxes[b]
        if truth_box is not None:
            truth  =  Variable(torch.zeros(len(truth_box),7)).cuda()
            truth[:,0  ] = b
            truth[:,1:5] = torch.from_numpy(truth_box).cuda()
            truth[:,5  ] = -1
            truth[:,6  ] =  1
            combine.append(truth)

        index = (proposals[:,0]==b).nonzero().view(-1)
        if len(index) !=0:
            proposal  = proposals[index]
            combine.append(proposal)

        if combine!=[]:
            sampled_proposal = torch.cat(combine, 0)
            sampled_proposals.append(sampled_proposal)

    sampled_proposals = torch.cat(sampled_proposals, 0)
    return sampled_proposals



# mask target ********************************************************************
#<todo> mask crop should match align kernel (same wait to handle non-integer pixel location (e.g. 23.5, 32.1))
def crop_instance(instance, box, size, threshold=0.5):
    H,W = instance.shape
    x0,y0,x1,y1 = np.rint(box).astype(np.int32)
    x0 = max(0,x0)
    y0 = max(0,y0)
    x1 = min(W,x1)
    y1 = min(H,y1)

    #<todo> filter this
    if 1:
        if x0==x1:
            x0=x0-1
            x1=x1+1
            x0 = max(0,x0)
            x1 = min(W,x1)
        if y0==y1:
            y0=y0-1
            y1=y1+1
            y0 = max(0,y0)
            y1 = min(H,y1)

    #print(x0,y0,x1,y1)
    crop = instance[y0:y1,x0:x1]
    crop = cv2.resize(crop,(size,size))
    crop = (crop>threshold).astype(np.float32)
    return crop




# cpu version
def make_one_mask_target(cfg, proposal, truth_box, truth_label, truth_instance):

    if truth_box is None: return None,None,None
    if proposal  is None: return None,None,None

    len_proposal = len(proposal)
    box = proposal[:,1:5]

    # overlaps: (rois x gt_boxes) -----
    overlaps = cython_box_overlap(box, truth_box)
    argmax_overlaps = np.argmax(overlaps,1)
    max_overlaps = overlaps[np.arange(len_proposal),argmax_overlaps]
    fg_inds = np.where( max_overlaps >= cfg.mask_train_fg_thresh_low)[0]

    #<todo> sampling for class balance
    if len(fg_inds)==0:
        raise NotImplementedError

    sampled_proposal = proposal[fg_inds]
    sampled_label = truth_label[argmax_overlaps[fg_inds]]
    sampled_instance = []
    for n in fg_inds:
        instance = truth_instance[argmax_overlaps[n]]
        box = proposal[n,1:5]
        crop = crop_instance(instance, box, cfg.mask_size)
        sampled_instance.append(crop[np.newaxis,:,:])

        #<debug>
        if 0:
            print(sampled_label[n])
            x0,y0,x1,y1 = box.astype(np.int32)
            image = (instance*255 ).astype(np.uint8)
            cv2.rectangle(image,(x0,y0),(x1,y1),128,1)
            image_show('image',image,2)
            image_show('crop',crop*255,2)
            cv2.waitKey(0)
    sampled_instance = np.vstack(sampled_instance)

    sampled_proposal = Variable(torch.from_numpy(sampled_proposal)).cuda()
    sampled_label    = Variable(torch.from_numpy(sampled_label)).cuda().long()
    sampled_instance = Variable(torch.from_numpy(sampled_instance)).cuda()
    return sampled_proposal, sampled_label, sampled_instance





def make_mask_target(cfg, proposals, truth_boxes, truth_labels, truth_instances):

    proposals = add_truth_boxes_to_proposals(cfg, proposals, truth_boxes)
    proposals = proposals.cpu().data.numpy()

    sampled_proposals  = []
    sampled_labels     = []
    sampled_instances  = []

    batch_size = len(truth_boxes)
    for b in range(batch_size):
        proposal  = proposals[np.where(proposals[:,0]==b)]
        truth_box = truth_boxes[b]
        truth_label = truth_labels[b]
        truth_instance = truth_instances[b]

        if truth_box is not None:
            sampled_proposal, sampled_label, sampled_instance = \
                make_one_mask_target(cfg, proposal, truth_box, truth_label, truth_instance)

            sampled_proposals.append(sampled_proposal)
            sampled_labels.append(sampled_label)
            sampled_instances.append(sampled_instance)

    sampled_proposals = torch.cat(sampled_proposals,0)
    sampled_labels    = torch.cat(sampled_labels,0)
    sampled_instances = torch.cat(sampled_instances,0)
    return sampled_proposals, sampled_labels, sampled_instances


#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()

 
 
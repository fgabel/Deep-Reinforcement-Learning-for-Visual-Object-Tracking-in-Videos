# reference:  https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/anchor_target_layer.py
from common import *
from model.mask_rcnn_lib.box import *
from model.mask_rcnn_lib.draw import *


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




# gpu version
## see https://github.com/ruotianluo/pytorch-faster-rcnn
def make_one_rcnn_target(cfg, proposal, truth_box, truth_label):

    if truth_box is None: return None,None,None
    if proposal  is None: return None,None,None

    num_classes      = cfg.num_classes
    num_proposal     = cfg.rcnn_train_batch_size
    num_fg_proposal  = int(np.round(cfg.rcnn_train_fg_fraction * num_proposal))

    len_proposal = len(proposal)
    box = proposal[:,1:5]


    # overlaps: (rois x gt_boxes) -----
    overlaps = cython_box_overlap(box, truth_box)
    argmax_overlaps = np.argmax(overlaps,1)
    max_overlaps = overlaps[np.arange(len_proposal),argmax_overlaps]

    fg_inds = np.where( max_overlaps >= cfg.rcnn_train_fg_thresh_low )[0]
    bg_inds = np.where((max_overlaps <  cfg.rcnn_train_bg_thresh_high) & \
                   (max_overlaps >= cfg.rcnn_train_bg_thresh_low))[0]


    # Small modification to the original version where we ensure a fixed number of regions are sampled
    # https://github.com/precedenceguo/mx-rcnn/commit/3853477d9155c1f340241c04de148166d146901d
    fg_inds_length = len(fg_inds)
    bg_inds_length = len(bg_inds)
    #print(fg_inds_length)

    if fg_inds_length > 0 and bg_inds_length > 0:
        num_fgs = min(num_fg_proposal, fg_inds_length)
        fg_inds = fg_inds[
            np.random.choice(fg_inds_length, size=num_fgs, replace=fg_inds_length<num_fgs)
        ]
        num_bgs = num_proposal - num_fgs
        bg_inds = bg_inds[
            np.random.choice(bg_inds_length, size=num_bgs, replace=bg_inds_length<num_bgs)
        ]

    elif fg_inds_length > 0:  #no bgs
        num_fgs = num_proposal
        num_bgs = 0
        fg_inds = fg_inds[
            np.random.choice(fg_inds_length, size=num_fgs, replace=fg_inds_length<num_fgs)
        ]

    elif bg_inds_length > 0:  #no fgs
        num_fgs = 0
        num_bgs = num_proposal
        bg_inds = bg_inds[
            np.random.choice(bg_inds_length, size=num_bgs, replace=bg_inds_length<num_bgs)
        ]
        num_fg_proposal = 0
    else:
        # no bgs and no fgs?
        # raise NotImplementedError
        num_fgs  = 0
        num_bgs  = num_proposal
        bg_inds  = np.random.choice(len_proposal, size=num_bgs, replace=len_proposal<num_bgs)


    assert ((num_fgs+num_bgs)== num_proposal)


    # selecting both fg and bg
    inds = np.concatenate([fg_inds, bg_inds], 0)
    sampled_proposal  = proposal[inds]

    #label
    sampled_label  = truth_label[argmax_overlaps][inds]
    if num_bgs>0:
        sampled_label[num_fgs:] = 0   # Clamp labels for the background RoIs to 0
    sampled_label = Variable(torch.from_numpy(sampled_label)).cuda().long()

    #target
    if num_fgs>0:
        target_truth_box = truth_box[argmax_overlaps][inds[:num_fgs]]
        target_box       = sampled_proposal[:num_fgs][:,1:5]
        target = box_transform(target_box, target_truth_box)
        sampled_target = Variable(torch.from_numpy(target)).cuda()
    else:
        sampled_target = None

    sampled_proposal = Variable(torch.from_numpy(sampled_proposal)).cuda()
    return sampled_proposal, sampled_label, sampled_target





def make_rcnn_target(cfg, proposals, truth_boxes, truth_labels):

    proposals = add_truth_boxes_to_proposals(cfg, proposals, truth_boxes)
    proposals = proposals.cpu().data.numpy()

    sampled_proposals = []
    sampled_labels    = []
    sampled_targets   = []

    batch_size = len(truth_boxes)
    for b in range(batch_size):
        proposal  = proposals[np.where(proposals[:,0]==b)]
        truth_box = truth_boxes[b]
        truth_label = truth_labels[b]

        if truth_box is not None:
            sampled_proposal, sampled_label, sampled_target = \
                make_one_rcnn_target(cfg, proposal, truth_box, truth_label)

            sampled_proposals.append(sampled_proposal)
            sampled_labels.append(sampled_label)
            sampled_targets.append(sampled_target)


    sampled_proposals = torch.cat(sampled_proposals,0)
    sampled_labels  = torch.cat(sampled_labels,0)
    sampled_targets = torch.cat(sampled_targets,0)
    return sampled_proposals, sampled_labels, sampled_targets


#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()

 
 
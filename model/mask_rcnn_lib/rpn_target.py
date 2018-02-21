# reference:  https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/anchor_target_layer.py

from dataset.reader import *

from model.mask_rcnn_lib.box import *
from model.mask_rcnn_lib.draw import *
from model.mask_rcnn_lib.rpn_nms import *





def draw_rpn_labels2d(label, label_weight, feature_widths, feature_heights, num_bases ):
    label        = label.cpu().data.numpy()
    label_weight = label_weight.cpu().data.numpy()

    label = label*128 + 127
    label_weight = label_weight*255

    label = label.astype(np.uint8)
    label_weight = label_weight.astype(np.uint8)

    labels        = unflat(label,        feature_widths, feature_heights, num_bases)
    label_weights = unflat(label_weight, feature_widths, feature_heights, num_bases)

    H,W = feature_widths[0], feature_heights[0]
    num_heads = len(feature_heights)
    for h in range(num_heads):
        labels[h] = cv2.resize(labels[h], (W,H), interpolation=cv2.INTER_LINEAR)
        label_weights[h] = cv2.resize(label_weights[h], (W,H), interpolation=cv2.INTER_LINEAR)

    labels = np.hstack(labels)
    label_weights = np.hstack(label_weights)


    return labels, label_weights


def draw_rpn_labels(image, windows, label, label_weight, is_fg=1, is_bg=1, is_print=1):
    image = image.copy()

    label        = label.cpu().data.numpy()
    label_weight = label_weight.cpu().data.numpy()
    windows = windows.astype(int)
    label   = label.astype(int)

    fg_inds = np.where(np.logical_and(label == 1,label_weight>0))[0]
    bg_inds = np.where(np.logical_and(label == 0,label_weight>0))[0]
    #fg_inds = np.where(label == 1)[0]
    #bg_inds = np.where(label == 0)[0]

    ## red  + dot : +ve label
    ## grey + dot : -ve label

    ## draw +ve/-ve labels ......
    num_windows   = len(windows)
    num_pos_label = len(fg_inds)
    num_neg_label = len(bg_inds)
    if is_print:
        print ('rpn label : num_pos=%d num_neg=%d,  all = %d'  %(num_pos_label, num_neg_label,num_pos_label+num_neg_label))

    if is_bg:
        for i in bg_inds:
            a = windows[i]
            cv2.rectangle(image,(a[0], a[1]), (a[2], a[3]), (64,64,64), 1)
            #cv2.circle(image,((a[0]+a[2])//2, (a[1]+a[3])//2),2, (64,64,64), -1, cv2.LINE_AA)
            cv2.circle(image,((a[0]+a[2])//2, (a[1]+a[3])//2),1, (0,0,255), -1)
    if is_fg:
        for i in fg_inds:
            a = windows[i]
            cv2.rectangle(image,(a[0], a[1]), (a[2], a[3]), (0,0,255), 1)
            #cv2.circle(image,((a[0]+a[2])//2, (a[1]+a[3])//2),2, (0,0,255), -1, cv2.LINE_AA)
            cv2.circle(image,((a[0]+a[2])//2, (a[1]+a[3])//2),1, (0,0,255), -1)

    return image


def draw_rpn_targets(image, windows, target, target_weight, is_before=1, is_after=1, is_print=1):
    image = image.copy()

    target = target.cpu().data.numpy()
    target_weight = target_weight.cpu().data.numpy()
    windows = windows.astype(int)

    target_inds = np.where(target_weight>0)[0]

    #draw +ve targets ......
    num_target = len(target_inds)
    if is_print:
        print ('rpn target : num_target=%d'  %(num_target))

    #is_before=0
    for i in target_inds:
        a = windows[i]
        t = target[i]
        b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
        b = b.reshape(-1).astype(np.int32)

        if is_before:
            cv2.rectangle(image,(a[0], a[1]), (a[2], a[3]), (0,0,255), 1)
            cv2.circle(image,((a[0]+a[2])//2, (a[1]+a[3])//2),2, (0,0,255), -1, cv2.LINE_AA)

        if is_after:
            cv2.rectangle(image,(b[0], b[1]), (b[2], b[3]), (0,255,255), 1)

    return image



# Faster-rcnn ground-truth layer rpn----------------------------------------

# gpu version
def make_one_rpn_target(cfg, windows, truth_box):
    num_windows  = len(windows)
    labels         = torch.zeros(num_windows).cuda().fill_(-1)
    label_weights  = torch.zeros(num_windows).cuda()
    targets        = torch.zeros((num_windows,4)).cuda()
    target_weights = torch.zeros(num_windows).cuda()
    if truth_box is not None:

        truth_box = torch.from_numpy(truth_box).float().cuda()
        num_truth_box = len(truth_box)

        # classification ---------------------------------------

        # overlaps between the windows and the gt
        overlaps = torch_box_overlap(windows, truth_box)
        max_overlaps, argmax_overlaps = overlaps.max(1)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        #  index = (a>b).nonzero().view(-1)
        #  mask  = (a>b)
        #
        bg_inds = (max_overlaps <  cfg.rpn_train_bg_thresh_high)
        fg_inds = (max_overlaps >= cfg.rpn_train_fg_thresh_low )
        labels[fg_inds]=1
        labels[bg_inds]=0

        truth_max_overlaps, truth_argmax_overlaps = overlaps.max(0)    # fg label: for each gt, window with highest overlap
        inds = (overlaps==truth_max_overlaps).nonzero()  # include multiple maxs
        fg_inds = inds[:,0]
        gt_inds = inds[:,1]
        labels[fg_inds]=1
        argmax_overlaps[fg_inds]=gt_inds


        #subsample
        fg_inds = (labels==1).nonzero().view(-1)
        bg_inds = (labels==0).nonzero().view(-1)
        fg_inds_length = len(fg_inds)
        bg_inds_length = len(bg_inds)

        #positive labels
        num_fgs = int(cfg.rpn_train_fg_fraction * cfg.rpn_train_batch_size)
        if fg_inds_length > num_fgs:
            fg_inds = fg_inds[
                torch.from_numpy(np.random.choice( fg_inds_length, size=num_fgs, replace=False)).long().cuda()
            ]
        else:
            num_fgs = fg_inds_length

        # negative labels
        num_bgs  = cfg.rpn_train_batch_size - num_fgs
        if bg_inds_length > num_bgs:
            bg_inds = bg_inds[
                torch.from_numpy(np.random.choice(bg_inds_length, size=num_bgs, replace=False)).long().cuda()
            ]
        else:
            num_bgs = bg_inds_length

        label_weights[fg_inds]=1
        label_weights[bg_inds]=1


        #regression----------------------------------------------
        target_inds = fg_inds
        target_windows   = windows[target_inds]
        target_truth_box = truth_box[argmax_overlaps[target_inds]]
        targets[target_inds] = torch_box_transform(target_windows, target_truth_box)
        target_weights[target_inds]=1

    # save
    labels          = Variable(labels)
    label_weights   = Variable(label_weights)
    targets         = Variable(targets)
    target_weights  = Variable(target_weights)
    return  labels, label_weights, targets, target_weights



###<todo> None gt cases
def make_rpn_target(cfg, windows, truth_boxes):

    windows = torch.from_numpy(windows).float().cuda()
    #<todo> preprocess e.g. remove boundary windows
    #<todo> preprocess e.g. remove small truth_boxes

    rpn_labels = []
    rpn_label_weights = []
    rpn_targets = []
    rpn_targets_weights = []

    batch_size = len(truth_boxes)
    for b in range(batch_size):
        truth_box = truth_boxes[b]

        rpn_label, rpn_label_weight, rpn_target, rpn_targets_weight = \
            make_one_rpn_target(cfg, windows, truth_box)

        rpn_labels.append(rpn_label.view(1,-1))
        rpn_label_weights.append(rpn_label_weight.view(1,-1))
        rpn_targets.append(rpn_target.view(1,-1,4))
        rpn_targets_weights.append(rpn_targets_weight.view(1,-1))


    rpn_labels          = torch.cat(rpn_labels, 0)
    rpn_label_weights   = torch.cat(rpn_label_weights, 0)
    rpn_targets         = torch.cat(rpn_targets, 0)
    rpn_targets_weights = torch.cat(rpn_targets_weights, 0)

    return rpn_labels, rpn_label_weights, rpn_targets, rpn_targets_weights


## check ############################################################################

def check_layer():
    image_id = '0402a81e75262469925ea893b6706183832e85324f7b1e08e634129f5d522cdd'

    dir = '/root/share/project/kaggle/science2018/data/image/stage1_train'
    image_file = dir + '/' + image_id + '/images/' + image_id + '.png'
    npy_file   = dir + '/' + image_id + '/label.npy'

    label = np.load(npy_file)
    image = cv2.imread(image_file,cv2.IMREAD_COLOR)

    batch_size =4
    images = []
    labels = []
    tensors = []
    truths  = []
    for b in range(batch_size):
        i, l = random_crop_transform2(image, label, 128, 128)
        images.append(i)
        labels.append(l)
        tensors.append(i.transpose((2,0,1))[np.newaxis,:,:,:])
        truths.append(l[np.newaxis,:,:])

        if 0:  #<debug>
            boxes = label_to_box(l)

            i1 = i.copy()
            l1 = ((l>0.5)[:,:,np.newaxis]*np.array((255,255,255))).astype(np.uint8)
            boxes=boxes.astype(np.int32)
            for box in boxes:
                x0,y0,x1,y1 = box
                cv2.rectangle(l1,(x0,y0),(x1,y1),(0,0,255),1)
                cv2.rectangle(i1,(x0,y0),(x1,y1),(0,0,255),1)

                image_show('l1',l1,2)
                image_show('i1',i1,2)
                cv2.waitKey(0)

    tensors = np.concatenate(tensors,0)
    truths  = np.concatenate(truths,0)
    tensors = torch.from_numpy(tensors).float().div(255)
    truths  = torch.from_numpy(truths).float()

    tensors = Variable(tensors)
    truths  = Variable(truths)

    #dummy features
    in_channels = 256
    feature_heights = [ 128, 64, 32, 16 ]
    feature_widths  = [ 128, 64, 32, 16 ]
    ps = []
    for height,width in zip(feature_heights,feature_widths):
        p = np.random.uniform(-1,1,size=(batch_size,in_channels,height,width)).astype(np.float32)
        p = Variable(torch.from_numpy(p)).cuda()
        ps.append(p)

    #------------------------

    # check layer
    cfg = type('', (object,), {})() #Configuration() #default configuration
    cfg.rpn_num_heads  = 4
    cfg.rpn_num_bases  = 3
    cfg.rpn_base_sizes = [ 8, 16, 32, 64 ]
    cfg.rpn_base_apsect_ratios = [1, 0.5,  2]
    cfg.rpn_strides    = [ 1,  2,  4,  8 ]

    cfg.rpn_train_batch_size     = 256  # rpn target
    cfg.rpn_train_fg_fraction    = 0.5
    cfg.rpn_train_bg_thresh_high = 0.3
    cfg.rpn_train_fg_thresh_low  = 0.7


    #start here --------------------------
    bases, windows = make_rpn_windows(cfg, ps)
    rpn_labels, rpn_label_weights, rpn_targets, rpn_targets_weights = make_rpn_target(cfg, windows, labels)


    for b in range(batch_size):
        rpn_label = rpn_labels[b]
        rpn_label_weight = rpn_label_weights[b]
        rpn_target = rpn_targets[b]
        rpn_target_weight = rpn_targets_weights[b]

        image = images[b]
        label = labels[b]
        gt_boxes = label_to_box(label)


        image1 = draw_gt_boxes(image, gt_boxes)
        #label2d, label_weight2d = draw_rpn_labels2d(rpn_label, rpn_label_weight, feature_widths, feature_heights, cfg.rpn_num_bases )
        image2 = draw_rpn_labels(image, windows, rpn_label, rpn_label_weight, is_fg=1, is_bg=1, is_print=1)
        image3 = draw_rpn_targets(image, windows, rpn_target, rpn_target_weight, is_before=1, is_after=1, is_print=1)


        image_show('image',image,3)
        image_show('label',label/label.max()*255,3)

        image_show('gt_boxes',image1,3)
        # image_show('label2d',label2d,3)
        # image_show('label_weight2d',label_weight2d,3)
        image_show('image2',image2,3)
        image_show('image3',image3,3)



        cv2.waitKey(0)



    #im_sh
#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()


 
 
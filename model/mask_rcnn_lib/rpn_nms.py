from common import *
import itertools

from model.mask_rcnn_lib.box import *
from model.mask_rcnn_lib.draw import *



#------------------------------------------------------------------------------
# make windows
def make_bases(base_size, base_apsect_ratios):
    bases = []
    for ratio in base_apsect_ratios:
        w = base_size/math.sqrt(ratio)
        h = base_size
        base =(-w//2, -h//2, w//2, h//2, )
        bases.append(base)

    bases = np.array(bases,np.float32)
    return bases


def make_windows(p, stride, bases):
    windows=[]
    _, _, H, W = p.size()
    for y, x in itertools.product(range(H),range(W)):
        cx = x*stride
        cy = y*stride
        for b in bases:
            windows.append(b + np.array([cx,cy,cx,cy],np.float32))

    windows = np.array(windows)
    return windows


def make_rpn_windows(cfg, ps):

    rpn_bases = []
    rpn_windows = []
    for i in range(cfg.rpn_num_heads):
        bases   = make_bases(cfg.rpn_base_sizes[i], cfg.rpn_base_apsect_ratios)
        windows = make_windows(ps[i], cfg.rpn_strides[i], bases)
        rpn_bases.append(bases)
        rpn_windows.append(windows)

    rpn_bases   = np.vstack(rpn_bases)
    rpn_windows = np.vstack(rpn_windows)
    return rpn_bases, rpn_windows

#------------------------------------------------------------------------------


# this is in gpu
def torch_rpn_nms(cfg, mode, inputs, probs_flat, deltas_flat, windows):

    if mode in ['train',]:
        nms_threshold  = cfg.rpn_train_nms_threshold
        nms_min_size   = cfg.rpn_train_nms_min_size
        nms_pre_top_n  = cfg.rpn_train_nms_pre_top_n
        nms_post_top_n = cfg.rpn_train_nms_post_top_n

    elif mode in ['eval', 'valid', 'test',]:
        nms_threshold  = cfg.rpn_test_nms_threshold
        nms_min_size   = cfg.rpn_test_nms_min_size
        nms_pre_top_n  = cfg.rpn_test_nms_pre_top_n
        nms_post_top_n = cfg.rpn_test_nms_post_top_n
    else:
        raise ValueError('rpn_nms(): invalid mode = %s?'%mode)


    probs  = probs_flat.detach().data
    deltas = deltas_flat.detach().data

    batch_size, num_windows = probs_flat.size()
    windows = torch.from_numpy(
                np.repeat(windows[np.newaxis,:,:],[batch_size],0)).cuda()

     # Convert windows into proposals via box transformations
    height, width = (inputs.size(2),inputs.size(3))  #original image width
    boxes = torch_box_transform_inv(windows.view(-1,4) , deltas.view(-1,4))
    boxes = torch_clip_boxes(boxes, width, height)
    boxes = boxes.view(batch_size,num_windows,4)

    # if nms_min_size[i] != -1: ##<todo>
    #     keep = torch_filter_boxes(boxes_i, nms_min_size[i])  #filter small size not implemented
    #     boxes_i = boxes_i [keep]
    #     probs_i = probs_i[keep]

    proposals = []
    for b in range(batch_size):
        ps = probs[b]
        bs = boxes[b]

        #https://github.com/pytorch/pytorch/pull/2329
        ps, idx = ps.sort(0, descending=True)
        if nms_pre_top_n > 0:
            idx = idx[:nms_pre_top_n]
            ps  = ps[:nms_pre_top_n].view(-1, 1)
            bs  = bs[idx, :]

        # Non-maximal suppression
        keep = torch_nms(torch.cat((bs, ps), 1), nms_threshold)
        if nms_post_top_n > 0:
            keep = keep[:nms_post_top_n]


        num_keeps = len(keep)
        proposal = torch.FloatTensor(num_keeps,7).cuda()
        proposal[:,0  ]=b
        proposal[:,1:5]=bs[keep]
        proposal[:,5  ]=ps[keep]
        proposal[:,6  ]=1  #class label

        proposals.append(Variable(proposal))

    proposals = torch.cat(proposals, 0)
    return proposals



#
# def filter_boxes(boxes, min_size):
#     '''Remove all boxes with any side smaller than min_size.'''
#     ws = boxes[:, 2] - boxes[:, 0] + 1
#     hs = boxes[:, 3] - boxes[:, 1] + 1
#     keep = np.where((ws >= min_size) & (hs >= min_size))[0]
#     return keep
#
# def torch_filter_boxes(boxes, min_size):
#     ws = boxes[:, 2] - boxes[:, 0] + 1
#     hs = boxes[:, 3] - boxes[:, 1] + 1
#     keep = (((ws >= min_size) + (hs >= min_size)) ==2).nonzero().view(-1)
#     return keep
#
#


# def draw_rpn_post_nms(image, proposals, top=100):
#
#     proposals = proposals.cpu().data.numpy().reshape(-1,5)
#     boxes  = proposals[:,:4]
#     scores = proposals[:,4]
#     num_proposals = len(proposals)
#
#     index = np.argsort(scores)     ##sort descend #[::-1]
#     if num_proposals>top: index = index[-top:]
#
#     num=len(index)
#     for n,i in enumerate(index):
#         box = boxes[i].astype(np.int)
#         v=255*n/num
#         color = (0,v,v)
#         cv2.rectangle(image,(box[0], box[1]), (box[2], box[3]), color, 1)
#
#
# #---------------------------------------------------------------------------
#
# #this is in gpu
# def torch_rpn_nms(x, probs_flat, deltas_flat, windows, cfg, mode='train'):
#
#     if mode in ['train',]:
#         nms_threshold  = cfg.rpn_train_nms_threshold
#         nms_min_size   = cfg.rpn_train_nms_min_size
#         nms_pre_top_n  = cfg.rpn_train_nms_pre_top_n
#         nms_post_top_n = cfg.rpn_train_nms_post_top_n
#
#     elif mode in ['eval', 'valid','test',]:
#         nms_threshold  = cfg.rpn_test_nms_threshold
#         nms_min_size   = cfg.rpn_test_nms_min_size
#         nms_pre_top_n  = cfg.rpn_test_nms_pre_top_n
#         nms_post_top_n = cfg.rpn_test_nms_post_top_n
#     else:
#         raise ValueError('rpn_nms(): invalid mode = %s?'%mode)
#
#     probs = []
#     boxes = []
#     height, width = (x.size(2),x.size(3))  #original image width
#     for i in range(cfg.rpn_num_heads):
#         windows_i = torch.from_numpy(windows[i]).cuda()
#         probs_flat_i  = probs_flat[i].detach().data
#         deltas_flat_i = deltas_flat[i].detach().data
#
#         # 1. Generate proposals from box deltas and windows (shifted bases)
#         probs_i  = probs_flat_i[:,1].contiguous().view(-1)
#         deltas_i = deltas_flat_i
#
#         # Convert windows into proposals via box transformations
#         boxes_i = torch_box_transform_inv(windows_i, deltas_i)
#         boxes_i = torch_clip_boxes(boxes_i, width, height)
#         if nms_min_size[i] != -1: ##<todo>
#             keep = torch_filter_boxes(boxes_i, nms_min_size[i])  #filter small size not implemented
#             boxes_i = boxes_i [keep]
#             probs_i = probs_i[keep]
#
#         probs.append(probs_i)
#         boxes.append(boxes_i)
#
#         # dump_dir='/root/share/project/ellen-object-detect/build/__reference__/mxnet_mask_rcnn/dump/pytorch'
#         # np.savetxt(dump_dir + '/scores_0.txt',scores_i.cpu().numpy().reshape(-1),fmt='%0.5f',delimiter='\t')
#         # np.savetxt(dump_dir + '/boxes_0.txt',boxes_i.cpu().numpy().reshape(-1,4),fmt='%0.5f',delimiter='\t')
#         # np.savetxt(dump_dir + '/windows_0.txt',windows[i].reshape(-1,4),fmt='%0.5f',delimiter='\t')
#         # exit(0)
#
#     # Pick the top region proposals
#     probs = torch.cat(probs)
#     boxes = torch.cat(boxes)
#
#     #https://github.com/pytorch/pytorch/pull/2329
#     probs, idx = probs.sort(0, descending=True)
#     if nms_pre_top_n > 0:
#         idx   = idx[:nms_pre_top_n]
#         probs = probs[:nms_pre_top_n].view(-1, 1)
#         boxes = boxes[idx, :]
#
#     # Non-maximal suppression
#     proposals = torch.cat((boxes, probs), 1)
#     keep = torch_nms(proposals, nms_threshold)
#     if nms_post_top_n > 0:
#         keep = keep[:nms_post_top_n]
#
#     proposals = proposals[keep, :]
#     proposals = Variable(proposals)
#     return proposals
#
# #
# # #this is in cpu:
# # def rpn_nms(x, scores_flat, deltas_flat, windows, inside_inds, cfg, mode='train'):
# #     scores_flat = scores_flat.detach()
# #     deltas_flat = deltas_flat.detach()
# #
# #     if mode=='train':
# #         nms_thresh    = cfg.rpn.train_nms_thresh
# #         nms_min_size  = cfg.rpn.train_nms_min_size
# #         nms_pre_topn  = cfg.rpn.train_nms_pre_topn
# #         nms_post_topn = cfg.rpn.train_nms_post_topn
# #
# #     elif mode=='eval':
# #         nms_thresh    = cfg.rpn.test_nms_thresh
# #         nms_min_size  = cfg.rpn.test_nms_min_size
# #         nms_pre_topn  = cfg.rpn.test_nms_pre_topn
# #         nms_post_topn = cfg.rpn.test_nms_post_topn
# #     else:
# #         raise ValueError('rpn_nms(): invalid mode = %s?'%mode)
# #
# #     height, width = (x.size(2),x.size(3))  #original image width
# #
# #     # 1. Generate proposals from box deltas and windows (shifted bases)
# #     scores_flat = scores_flat.view(-1, 2)
# #     scores_flat = F.softmax(scores_flat)  #[:,1].contiguous()
# #     scores_flat = scores_flat.cpu().data.numpy()
# #     scores  = scores_flat[inside_inds,1]
# #
# #     deltas_flat = deltas_flat.cpu().data.numpy()
# #     deltas  = deltas_flat[inside_inds]
# #     windows = windows[inside_inds]
# #
# #     # Convert anchors into proposals via box transformations
# #     proposals = box_transform_inv(windows, deltas)
# #
# #     # 2. clip predicted boxes to image
# #     proposals = clip_boxes(proposals, width, height)
# #
# #     # 3. remove predicted boxes with either height or width < threshold
# #     keep      = filter_boxes(proposals, nms_min_size)
# #     proposals = proposals[keep, :]
# #     scores    = scores[keep]
# #
# #     # 4. sort all (proposal, score) pairs by score from highest to lowest
# #     # 5. take top pre_nms_topN (e.g. 6000)
# #     order = scores.ravel().argsort()[::-1]
# #     if nms_pre_topn > 0:
# #         order = order[:nms_pre_topn]
# #         proposals = proposals[order, :]
# #         scores = scores[order]
# #
# #     # 6. apply nms (e.g. threshold = 0.7)
# #     # 7. take after_nms_topN (e.g. 300)
# #     # 8. return the top proposals
# #     keep = nms(np.hstack((proposals, scores)), nms_thresh)
# #     if nms_post_topn > 0:
# #         keep = keep[:nms_post_topn]
# #         proposals = proposals[keep, :]
# #         scores = scores[keep]
# #
# #     # Output rois blob
# #     # Our RPN implementation only supports a single input image, so all
# #     # batch inds are 0
# #     roi_scores=scores.squeeze()
# #
# #     num_proposals = len(proposals)
# #     inds = np.zeros((num_proposals, 1), dtype=np.float32)
# #     rois = np.hstack((inds, proposals))
# #
# #
# #     roi_scores = Variable(torch.from_numpy(roi_scores).type(torch.cuda.FloatTensor))
# #     rois       = Variable(torch.from_numpy(rois).type(torch.cuda.FloatTensor)) #i,x0,y0,x1,y1
# #
# #     return rois, roi_scores #<todo> modify roi return format later
#

#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



 
 

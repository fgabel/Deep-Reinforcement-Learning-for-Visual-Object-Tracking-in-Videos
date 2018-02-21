#  caffe-fast-rcnn/src/caffe/layers/smooth_L1_loss_layer.cu
#
#  sigma normlisation:
#     https://github.com/rbgirshick/py-faster-rcnn
#        see smooth_l1_loss_param { sigma: 3.0 }
#
#  std normlisation:
#        see cfg.TRAIN.BBOX_NORMALIZE_STDS



##-----------------------------------------------------------------
'''
http://pytorch.org/docs/0.1.12/_modules/torch/nn/modules/loss.html
Huber loss

class SmoothL1Loss(_Loss):


                          { 0.5 * (x_i - y_i)^2, if |x_i - y_i| < 1
    loss(x, y) = 1/n \sum {
                          { |x_i - y_i| - 0.5,   otherwise

    # loss = diff/(no._of_samples * dim_of_one_sample)
'''
#debug (cross check)
#l = modified_smooth_l1(rpn_deltas, rpn_targets, deltas_sigma)
##-----------------------------------------------------------------




from common import *

##  http://geek.csdn.net/news/detail/126833
def weighted_binary_cross_entropy_with_logits(logits, labels, weights):

    loss = weights*(logits.clamp(min=0) - logits*labels + torch.log(1 + torch.exp(-logits.abs())))
    loss = loss.sum()/(weights.sum()+1e-12)

    return loss


# original F1 smooth loss from rcnn
def weighted_smooth_l1( predicts, targets, weights, sigma=3.0):
    '''
        ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise

        inside_weights  = 1
        outside_weights = 1/num_examples
    '''

    predicts = predicts.view(-1)
    targets  = targets.view(-1)
    weights  = weights.view(-1)

    sigma2 = sigma * sigma
    diffs  =  predicts-targets
    smooth_l1_signs = torch.abs(diffs) <  (1.0 / sigma2)
    smooth_l1_signs = smooth_l1_signs.type(torch.cuda.FloatTensor)

    smooth_l1_option1 = 0.5 * diffs* diffs *  sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5  / sigma2
    loss = weights*(smooth_l1_option1*smooth_l1_signs + smooth_l1_option2*(1-smooth_l1_signs))

    loss = loss.sum()/(weights.sum()+1e-12)

    return loss


#---------------------------------------------------------------------------

def rpn_loss(logits_flat, deltas_flat, labels, label_weights,
             targets, target_weights,  deltas_sigma=3.0):

    batch_size, num_windows = logits_flat.size()
    target_weights = target_weights.view((batch_size, num_windows, 1)).expand((batch_size, num_windows, 4)).contiguous()

    rpn_cls_loss  = weighted_binary_cross_entropy_with_logits(logits_flat, labels, label_weights)
    rpn_reg_loss  = weighted_smooth_l1( deltas_flat, targets, target_weights, deltas_sigma)

    return rpn_cls_loss, rpn_reg_loss




# def check_layer1():
#
#     # set some dummy data
#     H = 5
#     W = 4
#     num_bases  = 3
#     batch_size = 1
#     L=8
#
#     scores_data = np.random.uniform(-1.,1.,(batch_size,num_bases*2,H,W))
#     deltas_data = np.random.uniform(-2.,2.,(batch_size,num_bases*4,H,W))
#
#     rpn_labels_data     = np.random.choice([0,1],L)
#     rpn_label_inds_data = np.random.choice(np.arange(H*W*num_bases),L, replace=False)
#
#     rpn_target_inds_data = rpn_label_inds_data[np.where(rpn_labels_data==1)[0]]
#     rpn_targets_data     = np.random.uniform(-2.,2.,(len(rpn_target_inds_data),4))
#
#     scores = Variable(torch.from_numpy(scores_data).type(torch.FloatTensor)).cuda()
#     deltas = Variable(torch.from_numpy(deltas_data).type(torch.FloatTensor)).cuda()
#     scores_flat = scores.permute(0, 2, 3, 1).contiguous().view(-1, 2)
#     deltas_flat = deltas.permute(0, 2, 3, 1).contiguous().view(-1, 4)
#
#     rpn_label_inds  = Variable(torch.from_numpy(rpn_label_inds_data).type(torch.cuda.LongTensor))
#     rpn_labels      = Variable(torch.from_numpy(rpn_labels_data).type(torch.cuda.LongTensor))
#     rpn_target_inds = Variable(torch.from_numpy(rpn_target_inds_data).type(torch.cuda.LongTensor))
#     rpn_targets     = Variable(torch.from_numpy(rpn_targets_data).type(torch.cuda.FloatTensor))
#
#
#
#     # check layer
#     rpn_cls_loss, rpn_reg_loss = rpn_loss(scores_flat, deltas_flat, rpn_label_inds, rpn_labels, rpn_target_inds, rpn_targets)
#
#
#     print(rpn_cls_loss)
#     print(rpn_reg_loss)
#     pass
#
#
# def check_layer():
#
#     DUMP_DIR = '/root/share/project/ellen-object-detect/results/dump/reference'
#     rpn_labels               = np.load(DUMP_DIR+'/loss__rpn_anchor_targets_labels.npy')  #torch.Size([1, 1, 333, 50])9*37,50
#     rpn_bbox_pred            = np.load(DUMP_DIR+'/loss__rpn_bbox_pred.npy')  #torch.Size([1, 37, 50, 36]) 1,37,50,9*4
#     rpn_bbox_targets         = np.load(DUMP_DIR+'/loss__rpn_bbox_targets.npy')
#     rpn_bbox_inside_weights  = np.load(DUMP_DIR+'/loss__rpn_bbox_inside_weights.npy')
#     rpn_bbox_outside_weights = np.load(DUMP_DIR+'/loss__rpn_bbox_outside_weights.npy')
#     # reference answer:
#     #1.00000e-02 *
#     #  6.2377
#
#     rpn_labels = rpn_labels.reshape((1,9,37,50)).transpose((0,2,3,1))
#     rpn_labels_flat = rpn_labels.reshape(-1)
#
#     target_inds = np.where(rpn_labels_flat==1)[0]
#     #array([10590, 11040, 11049, 11490, 11967, 12417, 12867, 13683, 14133, 14583])
#
#     rpn_bbox_pred_flat   = rpn_bbox_pred.reshape(-1,4)
#     rpn_bbox_targets_flat = rpn_bbox_targets.reshape(-1,4)
#     d = rpn_bbox_pred_flat[target_inds]
#     t = rpn_bbox_targets_flat[target_inds]
#
#
#     num_samples = 256
#     print('num_samples=%d'%num_samples)
#
#     deltas_sigma=3.0
#     deltas_sigma2 = deltas_sigma*deltas_sigma
#     dd = Variable(torch.from_numpy(d).type(torch.cuda.FloatTensor))
#     tt = Variable(torch.from_numpy(t).type(torch.cuda.FloatTensor))
#
#     rpn_reg_loss  = F.smooth_l1_loss( dd*deltas_sigma2, tt*deltas_sigma2, size_average=False)/deltas_sigma2/num_samples  #*4
#     print(rpn_reg_loss)
#     xx=0
#
#     pass

#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #check_layer()

 
 
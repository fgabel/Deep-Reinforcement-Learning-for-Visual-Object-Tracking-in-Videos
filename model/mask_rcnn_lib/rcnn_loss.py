from common import *
#---------------------------------------------------------------------------

# ##  https://discuss.pytorch.org/t/bceloss-from-scratch/2655/3
# #   https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
# def weighted_cross_entropy_with_logits(logits, labels, weights):
#
#     log_probs = F.log_softmax(logits)
#     labels    = labels.view(-1, 1)
#     loss = -torch.gather(log_probs, dim=1, index=labels)
#     loss = weights*loss.view(-1)
#
#     loss = loss.sum()/(weights.sum()+1e-12)
#     return loss
#
#
#
#
#
# # original F1 smooth loss from rcnn
# def weighted_smooth_l1( predicts, targets, weights, sigma=1.0):
#     '''
#         ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
#         SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
#                       |x| - 0.5 / sigma^2,    otherwise
#
#         inside_weights  = 1
#         outside_weights = 1/num_examples
#     '''
#
#     predicts = predicts.view(-1)
#     targets  = targets.view(-1)
#     weights  = weights.view(-1)
#
#     sigma2 = sigma * sigma
#     diffs  =  predicts-targets
#     smooth_l1_signs = torch.abs(diffs) <  (1.0 / sigma2)
#     smooth_l1_signs = smooth_l1_signs.type(torch.cuda.FloatTensor)
#
#     smooth_l1_option1 = 0.5 * diffs* diffs *  sigma2
#     smooth_l1_option2 = torch.abs(diffs) - 0.5  / sigma2
#     loss = weights*(smooth_l1_option1*smooth_l1_signs + smooth_l1_option2*(1-smooth_l1_signs))
#
#     loss = loss.sum()/(weights.sum()+1e-12)
#
#     return loss
#



## rcnn_loss uses deltas_sigma=1
def rcnn_loss(logits, deltas, labels, targets, deltas_sigma=1.0):

    batch_size, num_classes   = logits.size(0),logits.size(1)
    #label_weights = Variable(torch.ones((batch_size))).cuda()
    #rcnn_cls_loss = weighted_cross_entropy_with_logits(logits, labels, label_weights)
    rcnn_cls_loss = F.cross_entropy(logits, labels, size_average=True)

    num_pos = len(labels.nonzero())
    if num_pos > 0:
        # one hot encode
        select = Variable(torch.zeros((batch_size,num_classes))).cuda()
        select.scatter_(1, labels.view(-1,1), 1)
        select[:,0] = 0
        select = select.view(batch_size,num_classes,1).expand((batch_size,num_classes,4)).contiguous().byte()

        deltas = deltas.view(batch_size,num_classes,4)
        deltas = deltas[select].view(-1,4)

        deltas_sigma2 = deltas_sigma*deltas_sigma
        rcnn_reg_loss = F.smooth_l1_loss( deltas*deltas_sigma2, targets*deltas_sigma2,\
                                                 size_average=False)/deltas_sigma2/num_pos
    else:
        rcnn_reg_loss = Variable(torch.cuda.FloatTensor(1).zero_())

    return rcnn_cls_loss, rcnn_reg_loss



# def check_layer():
#
#     DUMP_DIR = '/root/share/project/ellen-object-detect/results/dump/reference'
#     rcnn_labels               = np.load(DUMP_DIR+'/loss__label.npy')  #torch.Size([256])
#     rcnn_bbox_pred            = np.load(DUMP_DIR+'/loss__bbox_pred.npy')  #torch.Size([256, 84]) 84=21*4
#     rcnn_bbox_targets         = np.load(DUMP_DIR+'/loss__bbox_targets.npy')
#     rcnn_bbox_inside_weights  = np.load(DUMP_DIR+'/loss__bbox_inside_weights.npy')
#     rcnn_bbox_outside_weights = np.load(DUMP_DIR+'/loss__bbox_outside_weights.npy')
#     # reference answer:
#     #  0.1032
#
#     num_classes = 21
#     num_samples = len(rcnn_labels)
#     print('num_samples=%d'%num_samples)
#
#     rcnn_bbox_pred_flat    = rcnn_bbox_pred.reshape(-1,4)
#     rcnn_bbox_targets_flat = rcnn_bbox_targets.reshape(-1,4)
#     rcnn_bbox_inside_weights_flat    = rcnn_bbox_inside_weights.reshape(-1,4)
#     rcnn_bbox_outside_weights_flat = rcnn_bbox_outside_weights.reshape(-1,4)
#
#     pos_inds = np.where(rcnn_labels!=0)[0]
#     rcnn_pos_labels = rcnn_labels[pos_inds]
#     pos_samples =len(rcnn_pos_labels)
#     print('pos_samples=%d'%pos_samples)
#
#     target_inds = np.arange(0,pos_samples)*num_classes + rcnn_pos_labels
#     d = rcnn_bbox_pred_flat[target_inds]
#     t = rcnn_bbox_targets_flat[target_inds]
#     w_in  = rcnn_bbox_inside_weights_flat[target_inds]
#     w_out = rcnn_bbox_outside_weights_flat[target_inds]
#
#     deltas_sigma=1.0
#     deltas_sigma2 = deltas_sigma*deltas_sigma
#     dd = Variable(torch.from_numpy(d).type(torch.cuda.FloatTensor))
#     tt = Variable(torch.from_numpy(t).type(torch.cuda.FloatTensor))
#
#     rcnn_reg_loss  = F.smooth_l1_loss( dd*deltas_sigma2, tt*deltas_sigma2, size_average=False)/deltas_sigma2/num_samples  #*4
#     print(rcnn_reg_loss)
#     #0.103212890625
#     xx=0
#
#     pass


#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()

 
 
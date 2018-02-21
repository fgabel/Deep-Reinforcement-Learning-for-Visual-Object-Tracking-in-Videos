from common import*
from model.configuration import*

#from model.roi_align_pool.module import RoIAlignMax as Crop
from model.mask_rcnn_lib.roi_align_pool.module import RoIAlignAvg as Crop
from model.mask_rcnn_lib.draw import *

from model.mask_rcnn_lib.rpn_nms import *
from model.mask_rcnn_lib.rpn_target import *
from model.mask_rcnn_lib.rpn_loss import *

from model.mask_rcnn_lib.rcnn_nms import *
from model.mask_rcnn_lib.rcnn_target import *
from model.mask_rcnn_lib.rcnn_loss import *

from model.mask_rcnn_lib.mask_nms import *
from model.mask_rcnn_lib.mask_target import *
from model.mask_rcnn_lib.mask_loss import *




############# resent50 pyramid feature net ##############################################################################

# class ConvBn2d(nn.Module):
#
#     def merge_bn(self):
#         raise NotImplementedError
#
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
#         super(ConvBn2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
#         self.bn   = nn.BatchNorm2d(out_channels)
#
#         if is_bn is False:
#             self.bn =None
#
#     def forward(self,x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         return x


## C layers ## ---------------------------

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, out_planes, is_downsample=False, stride=1):
        super(BottleneckBlock, self).__init__()
        self.is_downsample = is_downsample

        self.bn1   = nn.BatchNorm2d(in_planes,eps = 2e-5)
        self.conv1 = nn.Conv2d(in_planes,     planes, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes,eps = 2e-5)
        self.conv2 = nn.Conv2d(   planes,     planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn3   = nn.BatchNorm2d(planes,eps = 2e-5)
        self.conv3 = nn.Conv2d(   planes, out_planes, kernel_size=1, padding=0, stride=1, bias=False)

        if is_downsample:
            self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride, bias=False)


    def forward(self, x):
        if self.is_downsample:
            x = F.relu(self.bn1(x))
            z = self.conv1(x)
            z = F.relu(self.bn2(z))
            z = self.conv2(z)
            z = F.relu(self.bn3(z))
            z = self.conv3(z)
            z += self.downsample(x)
        else:
            z = F.relu(self.bn1(x))
            z = self.conv1(z)
            z = F.relu(self.bn2(z))
            z = self.conv2(z)
            z = F.relu(self.bn3(z))
            z = self.conv3(z)
            z += x

        return z


def make_layer_c0(in_planes, out_planes):
    layers = [
        nn.BatchNorm2d(in_planes, eps = 2e-5),
        nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=1, padding=2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)

def make_layer_c(in_planes, planes, out_planes, num_blocks, stride):
    layers = []
    layers.append(BottleneckBlock(in_planes, planes, out_planes, is_downsample=True, stride=stride))
    for i in range(1, num_blocks):
        layers.append(BottleneckBlock(out_planes, planes, out_planes))

    return nn.Sequential(*layers)



## P layers ## ---------------------------

class LateralBlock(nn.Module):
    def __init__(self, c_planes, p_planes, out_planes ):
        super(LateralBlock, self).__init__()
        self.lateral = nn.Conv2d(c_planes,  p_planes,   kernel_size=1, padding=0, stride=1)
        self.top     = nn.Conv2d(p_planes,  out_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, c , p):
        _,_,H,W = c.size()

        c = self.lateral(c)
        p = F.upsample(p, scale_factor=2,mode='nearest')
        p = p[:,:,:H,:W] + c
        p = self.top(p)

        return p


# 2 ways to downsize
# def make_layer_p5(in_planes, out_planes):
#     layers = [
#         nn.ReLU(inplace=True),
#         nn.Conv2d( in_planes, out_planes, kernel_size=3, stride=2, padding=1)
#     ]
#     return nn.Sequential(*layers)

def make_layer_p5(in_planes, out_planes):
    layers = [
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    ]
    return nn.Sequential(*layers)


## resenet50 + pyramid  ##
##  - indexing is different from paper. Paper is 1-index. Ours is 0-index.
##
class FeatureNet(nn.Module):

    def __init__(self, cfg, in_channels, out_channels=256 ):
        super(FeatureNet, self).__init__()
        self.cfg=cfg

        # bottom-top
        self.layer_c0 = make_layer_c0(in_channels, 64)
        self.layer_c1 = make_layer_c(   64,  64,  256, num_blocks=3, stride=2)  #out =  64*4 =  256
        self.layer_c2 = make_layer_c(  256, 128,  512, num_blocks=4, stride=2)  #out = 128*4 =  512
        self.layer_c3 = make_layer_c(  512, 256, 1024, num_blocks=6, stride=2)  #out = 256*4 = 1024
        self.layer_c4 = make_layer_c( 1024, 512, 2048, num_blocks=3, stride=2)  #out = 512*4 = 2048

        # top-down
        self.layer_p4 = nn.Conv2d   (2048, out_channels, kernel_size=1, stride=1, padding=0)
        self.layer_p3 = LateralBlock(1024, out_channels, out_channels)
        self.layer_p2 = LateralBlock( 512, out_channels, out_channels)
        self.layer_p1 = LateralBlock( 256, out_channels, out_channels)
        self.layer_p0 = LateralBlock(  64, out_channels, out_channels)


    def forward(self, x):
        pass                        #; print('input ',   x.size())
        c0  = self.layer_c0(x)      #
                                    #
        c1 = self.layer_c1(c0)      #; print('layer_c1 ',c1.size())
        c2 = self.layer_c2(c1)      #; print('layer_c2 ',c2.size())
        c3 = self.layer_c3(c2)      #; print('layer_c3 ',c3.size())
        c4 = self.layer_c4(c3)      #; print('layer_c4 ',c4.size())
                                    #
        p4 = self.layer_p4(c4)      #; print('layer_p4 ',p4.size())
        p3 = self.layer_p3(c3, p4)  #; print('layer_p3 ',p3.size())
        p2 = self.layer_p2(c2, p3)  #; print('layer_p2 ',p2.size())
        p1 = self.layer_p1(c1, p2)  #; print('layer_p1 ',p1.size())
        p0 = self.layer_p0(c0, p1)  #; print('layer_p0 ',p0.size())

        ps = [p0,p1,p2,p3]
        assert(self.cfg.rpn_num_heads == len(ps))

        return ps


    #-----------------------------------------------------------------------
    def load_pretrain_file(self,pretrain_file, skip=[]):
        raise NotImplementedError
    def merge_bn(self):
        raise NotImplementedError



############# various head ##############################################################################################

class RpnHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(RpnHead, self).__init__()
        self.num_heads = cfg.rpn_num_heads
        self.num_bases = cfg.rpn_num_bases
        self.sizes= None

        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.classify = nn.Conv2d(512, self.num_bases,   kernel_size=1, padding=0)
        self.delta    = nn.Conv2d(512, self.num_bases*4, kernel_size=1, padding=0)

    def forward(self, ps):
        self.sizes=[]
        for i in range(self.num_heads):
            self.sizes.append(ps[i].size())
        batch_size, C, H, W = self.sizes[0]

        deltas_flat = []
        logits_flat = []
        probs_flat  = []
        for i in range(self.num_heads):  # apply multibox head to feature maps
            p = ps[i]
            p = F.relu(self.conv(p))
            logit = self.classify(p)
            delta = self.delta(p)

            logit_flat = logit.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
            delta_flat = delta.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
            prob_flat  = F.sigmoid(logit_flat)

            logits_flat.append(logit_flat)
            deltas_flat.append(delta_flat)
            probs_flat.append(prob_flat)

        logits_flat = torch.cat(logits_flat,1)
        probs_flat  = torch.cat(probs_flat,1)
        deltas_flat = torch.cat(deltas_flat,1)

        return   logits_flat, probs_flat, deltas_flat


class CropRoi(nn.Module):
    def __init__(self, cfg):
        super(CropRoi, self).__init__()
        self.num_heads = cfg.rpn_num_heads
        self.pool_size = cfg.pool_size
        self.size_thresholds = cfg.rcnn_select_size_thresholds

        self.crops = nn.ModuleList()
        for i in range(cfg.rpn_num_heads):
            self.crops.append(
                Crop(cfg.pool_size, cfg.pool_size, 1/cfg.rpn_strides[i])
            )

    #proposal i,x0,y0,x1,y1,score
    #roi      i,x0,y0,x1,y1
    #box        x0,y0,x1,y1
    #det        x0,y0,x1,y1,score
    def forward(self, ps, proposals):
        num_proposals = len(proposals)

        ## this is  complicated. we need to decide for a given roi, which of the p0,p1, ..p3 layers to pool from
        boxes = proposals.detach().data[:,1:5]
        sizes = boxes[:,2:]-boxes[:,:2]
        sizes = torch.sqrt(sizes[:,0]*sizes[:,1])

        rois = proposals.detach().data[:,0:5]
        rois = Variable(rois)
        ids  = Variable(torch.arange(0,num_proposals, out=torch.LongTensor())).cuda()

        index = []
        crops = []
        for i in range(self.num_heads):
            threshold = self.size_thresholds[i]
            mask = Variable((threshold[0]<=sizes) * (sizes < threshold[1]))
            if mask.any() > 0:
                crops.append(
                    self.crops[i](ps[i], rois[mask.view(-1,1).expand_as(rois)].view(-1,5))
                )

                index.append(
                    ids[mask]
                )
        if index==[]: return None

        crops = torch.cat(crops,0)
        index = torch.sort(torch.cat(index,0))[1]
        crops = torch.index_select(crops,0,index)
        return crops


class RcnnHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(RcnnHead, self).__init__()
        num_classes = cfg.num_classes
        pool_size   = cfg.pool_size

        self.fc1 = nn.Linear(in_channels*pool_size*pool_size,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.classify = nn.Linear(1024,num_classes)
        self.delta    = nn.Linear(1024,num_classes*4)

    def forward(self, crops):

        x = crops.view(crops.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        logits = self.classify(x)
        deltas = self.delta(x)
        probs  = F.softmax(logits, dim=1)

        return logits, probs, deltas


class MaskHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(MaskHead, self).__init__()
        num_classes = cfg.num_classes

        self.conv1 = nn.Conv2d( in_channels,256, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(         256,256, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(         256,256, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(         256,256, kernel_size=3, padding=1, stride=1)
        #self.conv5 = nn.ConvTranspose2d(256,256, kernel_size=4, padding=1, stride=2, bias=False)
        self.segment = nn.Conv2d( 256,num_classes, kernel_size=1, padding=0, stride=1)


    def forward(self, crops):
        x = F.relu(self.conv1(crops),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        x = F.relu(self.conv3(x),inplace=True)
        x = self.conv4(x)
        logits = self.segment(x)
        probs  = F.sigmoid(logits)

        return logits, probs

############# mask rcnn net ##############################################################################

class MaskRcnnNet(nn.Module):

    def __init__(self, cfg):
        super(MaskRcnnNet, self).__init__()
        self.version = 'net version \'mask-rcnn-resnet50-fpn\''
        self.cfg  = cfg
        self.mode = 'train'

        feature_channels =256
        self.feature_net = FeatureNet(cfg, 3, feature_channels)
        self.rpn_head  = RpnHead (cfg,feature_channels)
        self.crop      = CropRoi (cfg)
        self.rcnn_head = RcnnHead(cfg,feature_channels)
        self.mask_head = MaskHead(cfg,feature_channels)



    #currently fast rcnn based detector only support single image  input
    def forward(self, inputs, truth_boxes=None,  truth_labels=None, truth_instances=None):
        cfg  = self.cfg
        mode = self.mode
        batch_size = len(inputs)

        #features
        features = data_parallel(self.feature_net, inputs)

        #rpn proposals
        self.rpn_logits_flat, self.rpn_probs_flat, self.rpn_deltas_flat = data_parallel(self.rpn_head, features)
        self.rpn_bases, self.rpn_windows = make_rpn_windows(cfg, features)
        self.rpn_proposals = torch_rpn_nms(cfg, mode, inputs, self.rpn_probs_flat, self.rpn_deltas_flat, self.rpn_windows)

        # detection
        if mode in ['train', 'valid']:
            self.rpn_labels, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights = \
                make_rpn_target(cfg, self.rpn_windows, truth_boxes )

            self.rpn_proposals, self.rcnn_labels, self.rcnn_targets,   = \
                make_rcnn_target(cfg, self.rpn_proposals, truth_boxes, truth_labels)


        rcnn_crops = self.crop(features, self.rpn_proposals)
        self.rcnn_logits, self.rcnn_probs, self.rcnn_deltas = data_parallel(self.rcnn_head, rcnn_crops)
        self.detections = rcnn_nms(cfg, mode, inputs, self.rpn_proposals,  self.rcnn_probs, self.rcnn_deltas)
        #self.masks = make_fake_masks(cfg, mode, inputs, self.detections)
        #return

        # segmentation
        self.rcnn_proposals = detections_to_proposals(self.detections)
        # <todo> decide to input proposal(from rpn) or detection(rcnn) for training
        if mode in ['train', 'valid']:
            self.rcnn_proposals, self.mask_labels, self.mask_instances,   = \
                make_mask_target(cfg,self.rcnn_proposals, truth_boxes, truth_labels, truth_instances)

            pass


        mask_crops = self.crop(features, self.rcnn_proposals)
        self.mask_logits, self.mask_probs = data_parallel(self.mask_head, mask_crops)  
        self.masks = mask_nms(cfg, mode, inputs, self.rcnn_proposals, self.mask_probs)  #<todo>
        #self.masks = self.mask_probs

        return #self.detections, self.masks  #return results in numpy


    def loss(self, inputs, truth_boxes, truth_labels, truth_instances):
        cfg  = self.cfg

        self.rpn_cls_loss, self.rpn_reg_loss = \
           rpn_loss( self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights)


        self.rcnn_cls_loss, self.rcnn_reg_loss = \
            rcnn_loss(self.rcnn_logits, self.rcnn_deltas, self.rcnn_labels, self.rcnn_targets)

        #self.mask_cls_loss = Variable(torch.zeros((1))).cuda()
        self.mask_cls_loss  = \
        	mask_loss(self.mask_logits, self.mask_labels, self.mask_instances )

        #self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss
        #self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss + self.rcnn_cls_loss + self.rcnn_reg_loss
        self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss + self.rcnn_cls_loss + self.rcnn_reg_loss + self.mask_cls_loss


        return self.total_loss


    #<todo> freeze bn for imagenet pretrain
    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


    def load_pretrain(self, pretrain_file):
        raise NotImplementedError







# check #################################################################
def run_check_feature_net():

    batch_size = 16
    C, H, W = 3, 128, 128
    feature_channels = 256

    x = torch.randn(batch_size,C,H,W)
    tensors = Variable(x).cuda()

    cfg = Configuration()
    feature_net = FeatureNet(cfg, in_channels=3, out_channels=256).cuda()

    ps = feature_net(tensors)

    print('')
    num_heads = len(ps)
    for i in range(num_heads):
        p = ps[i]
        print(i, p.size())



def run_check_rpn_head():

    batch_size  = 16
    in_channels = 256
    feature_heights = [ 128, 64, 32, 16 ]
    feature_widths  = [ 128, 64, 32, 16 ]

    ps = []
    for height,width in zip(feature_heights,feature_widths):
        p = np.random.uniform(-1,1,size=(batch_size,in_channels,height,width)).astype(np.float32)
        p = Variable(torch.from_numpy(p)).cuda()
        ps.append(p)


    cfg = Configuration()
    rpn_head = RpnHead(cfg, in_channels).cuda()

    logits_flat, probs_flat, deltas_flat = rpn_head(ps)

    print('logits_flat ',logits_flat.size())
    print('probs_flat  ',probs_flat.size())
    print('deltas_flat ',deltas_flat.size())
    print('')

    num_heads = rpn_head.num_heads
    for i in range(num_heads):
        print(i, rpn_head.sizes[i])



#<todo> check this later ...
def run_check_crop_head():


    #proposal i,x0,y0,x1,y1,score, label
    batch_size = 4
    num_proposals = 8
    xs = np.random.randint(0,64,num_proposals)
    ys = np.random.randint(0,64,num_proposals)
    sizes  = np.random.randint(8,64,num_proposals)
    scores = np.random.uniform(0,1,num_proposals)

    proposals = np.zeros((num_proposals,7),np.float32)
    proposals[:,0] = np.random.choice(batch_size,num_proposals)
    proposals[:,1] = xs
    proposals[:,2] = ys
    proposals[:,3] = xs+sizes
    proposals[:,4] = ys+sizes
    proposals[:,5] = scores
    proposals[:,6] = 1

    index = np.argsort(-scores)  #descending
    proposals = proposals[index]
    proposals= Variable(torch.from_numpy(proposals)).cuda()


    #feature maps
    batch_size = 8
    in_channels = 4
    feature_heights = [ 128, 64, 32, 16 ]
    feature_widths  = [ 128, 64, 32, 16 ]

    ps = []
    for i, (H,W) in enumerate(zip(feature_heights,feature_widths)):
        p = np.zeros((batch_size,in_channels,H,W),np.float32)
        for b in range(batch_size):
            for y in range(H):
                for x in range(W):
                    p[b,0,y,x]=y
                    p[b,1,y,x]=x
                    p[b,2,y,x]=b
                    p[b,3,y,x]=i
        p = Variable(torch.from_numpy(p)).cuda()
        ps.append(p)


    #--------------------------------------
    cfg = Configuration()
    crop_net = CropRoi(cfg).cuda()
    crops = crop_net(ps, proposals)

    print('crops', crops.size())
    #exit(0)

    crops = crops.data.cpu().numpy()
    proposals = proposals.data.cpu().numpy()
    #for m in range(num_proposals):
    for m in range(8):
        crop = crops[m]
        proposal = proposals[m]

        i,x0,y0,x1,y1,score,label = proposal

        print ('i=%d, x0=%3d, y0=%3d, x1=%3d, y1=%3d, score=%0.2f'%(i,x0,y0,x1,y1,score) )
        print (crop )
        print ('')


def run_check_rcnn_head():

    num_rois     = 100
    in_channels  = 256
    pool_size    = 16

    crops = np.random.uniform(-1,1,size=(num_rois,in_channels, pool_size,pool_size)).astype(np.float32)
    crops = Variable(torch.from_numpy(crops)).cuda()

    cfg = Configuration()
    assert(pool_size==cfg.pool_size)

    rcnn_head = RcnnHead(cfg,in_channels).cuda()


    logits, probs, deltas = rcnn_head(crops)

    print('logits ',logits.size())
    print('probs  ',probs.size())
    print('deltas ',deltas.size())
    print('')



def run_check_mask_head():

    num_rois    = 100
    in_channels = 256
    pool_size = 16


    crops = np.random.uniform(-1,1,size=(num_rois, in_channels, pool_size, pool_size)).astype(np.float32)
    crops = Variable(torch.from_numpy(crops)).cuda()

    cfg = Configuration()
    assert(pool_size==cfg.pool_size)

    mask_head = MaskHead(cfg, in_channels).cuda()
    mask_head

    logits, probs = mask_head(crops)

    print('logits ',logits.size())
    print('')




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # run_check_feature_net()
    # run_check_rpn_head()
    run_check_crop_head()
    # run_check_rcnn_head()
    # run_check_mask_head()
    # run_check_mask_rcnn_net()

    #print(feature_net)
    #print(x)



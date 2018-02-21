import sys, os
sys.path.append(os.path.dirname(__file__))

from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d
from function import RoIAlignFunction


class RoIAlign(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width,
                                self.spatial_scale)(features, rois)

class RoIAlignAvg(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x =  RoIAlignFunction(self.aligned_height+1, self.aligned_width+1,
                                self.spatial_scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'aligned_width=' + str(self.aligned_width) \
            + ', aligned_height=' + str(self.aligned_height) \
            + ', spatial_scale=' + str(self.spatial_scale) \
            + ')'

class RoIAlignMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x =  RoIAlignFunction(self.aligned_height+1, self.aligned_width+1,
                                self.spatial_scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'aligned_width=' + str(self.aligned_width) \
            + ', aligned_height=' + str(self.aligned_height) \
            + ', spatial_scale=' + str(self.spatial_scale) \
            + ')'
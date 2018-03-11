# edit settings here
ROOT_DIR =''



DATA_DIR    = '' 
RESULTS_DIR = ROOT_DIR + '/results'

##---------------------------------------------------------------------
import os
import copy
from datetime import datetime
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#numerical libs
import math
import numpy as np
import random
import PIL
from PIL import Image # import jpg in python
import cv2

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg')


# torch libs
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.parallel.data_parallel import data_parallel
from torch.utils.data import Dataset
import torchvision


# std libs
import collections
import numbers
import inspect
import shutil
from timeit import default_timer as timer
from __future__ import print_function, division



import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time
import matplotlib.pyplot as plt

import skimage
import skimage.color
from skimage import io, transform
# import all images from a folder, see the dataloader
from skimage.io import imread_collection, imread, concatenate_images 
from scipy import ndimage
from shapely.geometry import Polygon # for the Polygon



#---------------------------------------------------------------------------------
print('@%s:  ' % os.path.basename(__file__))

if 1:
    SEED=35202#1510302253  #int(time.time()) #
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print ('\tset random seed')
    print ('\t\tSEED=%d'%SEED)

if 1:
    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True
    print ('\tset cuda environment')
    #print ('\t\ttorch.__version__              =', torch.__version__)
    #print ('\t\ttorch.version.cuda             =', torch.version.cuda)
    print ('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
    try:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']  =',os.environ['CUDA_VISIBLE_DEVICES'])
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']  =','None')
        NUM_CUDA_DEVICES = 1

    print ('\t\ttorch.cuda.device_count()   =', torch.cuda.device_count())
    print ('\t\ttorch.cuda.current_device() =', torch.cuda.current_device())


print('')

#---------------------------------------------------------------------------------
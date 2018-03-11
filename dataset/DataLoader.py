<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:44:42 2018

@author: Einmal
"""

# Build a dataset loader according to 
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from __future__ import print_function, division

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from skimage import io, transform
import torchvision
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import time
import os
import copy
from PIL import Image # import jpg in python
from skimage.io import imread_collection, imread, concatenate_images # import all images from a folder, see the dataloader




#%%

# Load the list of all videos
vid_list = pd.read_csv('F:/vot2017/list.txt', header = None)

# Name of a video can be accessed by e.g. vid_list[0][5]
print( vid_list[0][33])

 # Nr. of videos available
n = vid_list.shape[0]


#%% Test drawing

# Load the gt boxes, which are apparently stored in coordinates of 4 points
gt = pd.read_csv('F:/vot2017/ants1/groundtruth.txt', header = None)


# Transform dataset into array
# Not neccessary dt.iloc[x] does it

#im  = Image.open('F:/vot2017/ants1/00000092.jpg')

# Draws a rectangle given the coordinates of all four corners in one array
# Where the order is upper-left, upper-right, lower-rigth, lower-left
def draw_gt(im, coords):
    """ Arguments:
            im = image
            coords = coords of all corners as in ground truth files(u.l,u.r,l.r,l.l)(u=upper,l = lower)
        """
    plt.imshow(im)
    Xs = coords[::2] # Save Xcoords
    Ys = coords[1::2] # Save Ycoords
    for i in range(4):
        if i < 3:
            plt.plot([Xs[i],Xs[i+1]],[Ys[i],Ys[i+1]],'k-', color = 'r',lw=1)
        elif i == 3:
            plt.plot([Xs[i],Xs[0]],[Ys[i],Ys[0]],'k-', color ='r', lw=1)
    plt.show()


#draw_gt(im, gt.iloc[91])
#Check



#%% Dataset class
    
'''Output is the imagesequence in an np.array format and the gt aswell.'''

class VOT2017_dataset(Dataset):
    """This is the VOT2017 dataset"""
    def __init__(self, csv_file, root_dir, transform = None):
        """ Arguments:
            csv_file(string): Path to list file, where all videos are listed
            root_dir(string): Directory with all the videos
            transform(callable, optional): Will transform on a sample(for pytorch I guess)
        """
        self.vot_list = pd.read_csv(csv_file, header = None)
        self.root_dir = root_dir
        self.transform = transform
    
    # Returns the nr of videos available
    def __len__(self):
        return len(self.vot_list)
    
    # Return the complete video sequence
    def __getitem__(self, vid_idx):
        """ Arguments:
            vid_idx(int): Video Index to be fetched form the video list
        """
        vid_name_path = os.path.join(self.root_dir, 
                                     self.vot_list.iloc[vid_idx,0],
                                     '*.jpg')
        
        gt = pd.read_csv(os.path.join(self.root_dir, 
                                      self.vot_list.iloc[vid_idx,0],
                                      'groundtruth.txt'), header = None)
        
        im_seq = imread_collection(vid_name_path)
        
        # Image collection to np.array
        images = concatenate_images(im_seq) # Shape(Nr. of images, h, w, RGB)
        
        # Also convert the gt to np.array
        gt = gt.values
        
        sample = {'Video': images, 'gt': gt}
        
        # Cant tell yet what this is for
        if self.transform:
            sample = self.transform(sample)    
        return sample
    

#%% Test the dataset class

test = VOT2017_dataset(csv_file= 'F:/vot2017/list.txt',
                       root_dir= 'F:/vot2017/')  

# E.g. load the second video of the vid_list
sample = test[2]

# Simply draw a single video - here the idx refers to the image in the sequence
draw_gt(sample['Video'][0], sample['gt'][0])


#%% Just for information - Find the smallest sized video

Vids = vid_list.shape[0]

Size = np.zeros((Vids, 2))

for i in range(Vids):
    im = Image.open(os.path.join('F:/vot2017/', 
                                     vid_list.iloc[i,0],
                                     '00000001.jpg'))
    Size[i,0] = im.size[0]
    Size[i,1] = im.size[1]

# Smallest size ist 320 x240
#Histogram of image sizes
plt.hist(Size)




#%% Transforms - Rescale/Resize
    
# Rescaling of an image so that we can feed it with the same size into a network
# Also the groundtruth boxes have to be rescaled accordingly!
# Problem atm rescale the whole Video!! - so far only with for loop

class Rescale(object):
    '''
    Rescale the image in a sample to a given size
    
    Arguments: output_size(tuple): Desired output size
               idx(int) : For now idx of the image to be resized
    '''
    
    # Check if output_size is a tuple,
    # maybe also assert if it isnt bigger than the smallest image?
    def __init__(self, output_size):
        assert isinstance(output_size,(tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        # Split the sample in video and gt
        images, gt = sample['Video'], sample['gt']
        nr = len(images) # Save the amount of images to iterate over
        print(nr)
        # Save heigth and width of video
        h, w = images.shape[1:3] # heigth and width are the 2nd and 3rd entry
        
        new_h, new_w = self.output_size
        
        
        # I dont like this part due to the for loop.! 
        # Initialize the resized image sequence array
        img = np.zeros((nr,new_h,new_w, images.shape[3]))
        # Iterate over all images and resize them to the given scale.
        for i in range(nr):
            img[i,:,:,:] = transform.resize(images[i,:,:,:], (new_h, new_w))
        
    
        # Here the groundtruth boxes are rescaled aswell
        gt_new = gt*np.array((new_w/w, new_h/h, new_w/w,new_h/h, new_w/w,new_h/h, new_w/w, new_h/h))
        
        return {'Video': img, 'gt': gt_new}



#%% Test rescaling
        
scale = Rescale((220,280))


transformed_sample = scale(sample)
draw_gt(transformed_sample['Video'][100], transformed_sample['gt'][100])

# Check

#%% Transforms - ToTensor

# Transform the loaded image collection to Tensors

class ToTensor(object):
    '''Convert sample to tensor'''
    def __call__(self, sample):
        # Load the sample and split it
        images, gt = sample['Video'], sample['gt']
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # How does this relate to videos/imagesequences?
        images = images.transpose((0,3,1,2))
        
        return {'Video': torch.from_numpy(images),
                'gt': torch.from_numpy(gt)}
        
#%% Test the ToTensor
        
tens = ToTensor()
tens(sample)

# Still have to test this
=======
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:44:42 2018

@author: Einmal
"""

# Build a dataset loader according to 
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from __future__ import print_function, division

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from skimage import io, transform
import torchvision
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import time
import os
import glob
import copy
from PIL import Image # import jpg in python
from skimage.io import imread_collection, imread, concatenate_images # import all images from a folder, see the dataloader
import shapely

#%%

# Load the list of all videos
vid_list = pd.read_csv('F:/vot2017/list.txt', header = None)

# Name of a video can be accessed by e.g. vid_list[0][5]
print( vid_list[0][33])

 # Nr. of videos available
n = vid_list.shape[0]


#%% Test

test =os.path.join('F:/vot2017/',
                   vid_list.iloc[0,0])

included_extension = ['jpg']
file_names = [fn for fn in os.listdir(test)
              if any(fn.endswith(ext) for ext in included_extension)]
#%% Is rectangular 

#for i in range(vid_list.shape[0]):
    gt = pd.read_csv(os.path.join('F:/vot2017/', 
                              vid_list.iloc[3,0],
                              'groundtruth.txt'), header = None)
    gt = gt.values
    Xs = gt[:,::2] # Save Xcoords
    Ys = gt[:,1::2] # Save Ycoords
    print(Xs)
#%% Test drawing

# Load the gt boxes, which are apparently stored in coordinates of 4 points
gt = pd.read_csv('F:/vot2017/ball1/groundtruth.txt', header = None)
gt = gt.values

# Transform dataset into array
# Not neccessary dt.iloc[x] does it

im  = Image.open('F:/vot2017/ball1/00000095.jpg')

# Draws a rectangle given the coordinates of all four corners in one array
# Where the order is upper-left, upper-right, lower-rigth, lower-left
def draw_gt(im, coords):
    """ Arguments:
            im = image
            coords = coords of all corners as in ground truth files(u.l,u.r,l.r,l.l)(u=upper,l = lower)
        """
    plt.imshow(im)
    Xs = coords[::2] # Save Xcoords
    Ys = coords[1::2] # Save Ycoords
    for i in range(4):
        if i < 3:
            plt.plot([Xs[i],Xs[i+1]],[Ys[i],Ys[i+1]],'k-', color = 'r',lw=1)
        elif i == 3:
            plt.plot([Xs[i],Xs[0]],[Ys[i],Ys[0]],'k-', color ='r', lw=1)
    plt.show()


draw_gt(im, gt[94])
#Check



#%% Dataset class
    
'''Output is the imagesequence in an np.array format and the gt aswell.'''

class VOT2017_dataset(Dataset):
    """This is the VOT2017 dataset"""
    def __init__(self, csv_file, root_dir, transform = None):
        """ Arguments:
            csv_file(string): Path to list file, where all videos are listed
            root_dir(string): Directory with all the videos
            transform(callable, optional): Will transform on a sample(for pytorch I guess)
            
        """
        self.vot_list = pd.read_csv(csv_file, header = None)
        self.root_dir = root_dir
        self.transform = transform
    
    
    # Return the complete video sequence
    def __getitem__(self, vid_idx, T = 10):
        """ Arguments:
            vid_idx(int): Video Index to be fetched form the video list
            T(int): Nr of Images in sequence - default == 10
        """
        gt = pd.read_csv(os.path.join(self.root_dir, 
                                      self.vot_list.iloc[vid_idx,0],
                                      'groundtruth.txt'), header = None)
        
        vid_name_path = os.path.join(self.root_dir, 
                                     self.vot_list.iloc[vid_idx,0],
                                     '*.jpg')
        
        file_names = glob.glob(vid_name_path)
        
        rand_start = np.random.randint(0, len(file_names)-T+1)
        
        file_names = file_names[rand_start:(rand_start+T-1)]
        
        im_seq = imread_collection(file_names)
        
        # Image collection to np.array
        images = concatenate_images(im_seq) # Shape(Nr. of images, h, w, RGB)
        
        # Also convert the gt to np.array
        gt = gt.values
        gt = gt[rand_start:(rand_start+T-1),:]
        
        sample = {'Video': images, 'gt': gt}
        
        # Cant tell yet what this is for
        if self.transform:
            sample = self.transform(sample)    
        return sample
    

#%% Test the dataset class

test = VOT2017_dataset(csv_file= 'F:/vot2017/list.txt',
                       root_dir= 'F:/vot2017/')  

# E.g. load a of the vid_list
sample = test.__getitem__(0, T = 20)

# Simply draw a single video - here the idx refers to the image in the sequence
draw_gt(sample['Video'][10], sample['gt'][10])

#%% define loss/reward functions; given the coordinates of all corners

def loss_v1(pred, gt):
    r = - np.mean(np.absolute(pred-gt)) - np.max(np.absolute(pred-gt))
    return r


# Calculate the reward given all the for corners x1,y1,x2,y2,x3,y3,x4,y4
def loss_v2(pred, gt):
    #reorder the coord in tuples for the polygon
    pred_re = [(pred[0],pred[1]),(pred[2],pred[3]), (pred[4],pred[5]),(pred[6],pred[7])]
    gt_re   = [(gt[0],gt[1]),(gt[2],gt[3]),(gt[4],gt[5]),(gt[6],gt[7])]
    
    pred_poly = Polygon(pred_re)
    gt_poly = Polygon(gt_re)
    # Reward == Intersection/total area
    r = pred_poly.intersection(gt_poly).area/(pred_poly.area + gt_poly.area - pred_poly.intersection(gt_poly).area)
    return r
#%% Test reward functions
    
test_1 = np.array((0,0,0,1,1,1,1,0))
test_2 = np.array((0.5,0,0.5,1,1.5,1,1.5,0))

print(loss_v1(test_1, test_2))
print(loss_v2(test_1, test_2))


#%% Just for information - Find the smallest sized video

Vids = vid_list.shape[0]

Size = np.zeros((Vids, 2))

for i in range(Vids):
    im = Image.open(os.path.join('F:/vot2017/', 
                                    vid_list.iloc[i,0],
                                     '00000001.jpg'))
    Size[i,0] = im.size[0]
    Size[i,1] = im.size[1]

# Smallest size ist 320 x240
#Histogram of image sizes
plt.hist(Size)




#%% Transforms - Rescale/Resize
    
# Rescaling of an image so that we can feed it with the same size into a network
# Also the groundtruth boxes have to be rescaled accordingly!
# Problem atm rescale the whole Video!! - so far only with for loop

class Rescale(object):
    '''
    Rescale the image in a sample to a given size
    
    Arguments: output_size(tuple): Desired output size
               idx(int) : For now idx of the image to be resized
    '''
    
    # Check if output_size is a tuple,
    # maybe also assert if it isnt bigger than the smallest image?
    def __init__(self, output_size):
        assert isinstance(output_size,(tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        # Split the sample in video and gt
        images, gt = sample['Video'], sample['gt']
        nr = len(images) # Save the amount of images to iterate over
        print(nr)
        # Save heigth and width of video
        h, w = images.shape[1:3] # heigth and width are the 2nd and 3rd entry
        
        new_h, new_w = self.output_size
        
        
        # I dont like this part due to the for loop.! 
        # Initialize the resized image sequence array
        img = np.zeros((nr,new_h,new_w, images.shape[3]))
        # Iterate over all images and resize them to the given scale.
        for i in range(nr):
            img[i,:,:,:] = transform.resize(images[i,:,:,:], (new_h, new_w))
        
    
        # Here the groundtruth boxes are rescaled aswell
        gt_new = gt*np.array((new_w/w, new_h/h, new_w/w,new_h/h, new_w/w,new_h/h, new_w/w, new_h/h))
        
        return {'Video': img, 'gt': gt_new}



#%% Test rescaling
        
scale = Rescale((220,280))


transformed_sample = scale(sample)
draw_gt(transformed_sample['Video'][100], transformed_sample['gt'][100])

# Check

#%% Transforms - ToTensor

# Transform the loaded image collection to Tensors

class ToTensor(object):
    '''Convert sample to tensor'''
    def __call__(self, sample):
        # Load the sample and split it
        images, gt = sample['Video'], sample['gt']
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # How does this relate to videos/imagesequences?
        images = images.transpose((0,3,1,2))
        
        return {'Video': torch.from_numpy(images),
                'gt': torch.from_numpy(gt)}
        
#%% Test the ToTensor


rescale = Rescale()
rescale(sample)
tens = ToTensor()
tens(sample)

# Apparently not enoguh RAM on my machine here to fully check that it works :D
>>>>>>> b297fdf775d26e7df2deb91721412a875e22dd77

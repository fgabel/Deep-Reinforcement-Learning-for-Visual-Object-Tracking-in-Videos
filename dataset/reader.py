from common import *

from dataset.transform import *
from dataset.sampler import *
from utility.file import *
from utility.draw import *


#data reader  ----------------------------------------------------------------
class ScienceDataset(Dataset):

    def __init__(self, split, transform=None, mode='train'):
        super(ScienceDataset, self).__init__()
        start = timer()

        self.split = split
        self.transform = transform
        self.mode = mode

        #read split
        ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

        #save
        self.ids = ids

        #print
        print('\ttime = %0.2f min'%((timer() - start) / 60))
        print('\tnum_ids = %d'%(len(self.ids)))
        print('')


    def __getitem__(self, index):
        id = self.ids[index]
        image_id = id.split('/')[-1]
        image = cv2.imread(DATA_DIR + '/image/' + id + '/images/' + image_id +'.png', cv2.IMREAD_COLOR)

        if self.mode in ['train']:
            multi_mask = np.load( DATA_DIR + '/image/' + id + '/multi_mask.npy')#.astype(int32)

            if self.transform is not None:
                return self.transform(image, multi_mask, index)
            else:
                return input, multi_mask, index

        if self.mode in ['test']:
            if self.transform is not None:
                return self.transform(image,index)
            else:
                return image, index

    def __len__(self):
        return len(self.ids)
# draw  ----------------------------------------------------------------
def multi_mask_to_overlay(multi_mask):
    overlay = skimage.color.label2rgb(multi_mask, bg_label=0, bg_color=(0, 0, 0))*255
    overlay = overlay.astype(np.uint8)
    return overlay


# modifier  ----------------------------------------------------------------
def thresh_to_inner_contour(thresh):
    thresh_pad = np.lib.pad(thresh, ((1, 1), (1, 1)), 'reflect')
    contour = thresh_pad[1:-1,1:-1] & (
            (thresh_pad[1:-1,1:-1] != thresh_pad[:-2,1:-1]) \
          | (thresh_pad[1:-1,1:-1] != thresh_pad[2:,1:-1])  \
          | (thresh_pad[1:-1,1:-1] != thresh_pad[1:-1,:-2]) \
          | (thresh_pad[1:-1,1:-1] != thresh_pad[1:-1,2:])
    )
    return contour


def multi_mask_to_annotation(multi_mask):
    H,W = multi_mask.shape[:2]
    count = multi_mask.max()

    box      = []
    label    = []
    instance = []
    for i in range(count):
        thresh = (multi_mask==(i+1))
        if thresh.sum()>1:
            #<todo> filter small, etc

            y,x = np.where(thresh)
            y0 = y.min()
            y1 = y.max()
            x0 = x.min()
            x1 = x.max()
            w = (x1-x0)+1
            h = (y1-y0)+1

            #f  = int(0.3*min(w,h))
            border  = int(0.3*(w+h)/2)
            x0 = x0-border
            x1 = x1+border
            y0 = y0-border
            y1 = y1+border

            #clip
            x0 = max(0,x0)
            y0 = max(0,y0)
            x1 = min(W-1,x1)
            y1 = min(H-1,y1)

            #<todo> filter small
            box.append([x0,y0,x1,y1])
            label.append(1) #<todo> now assume one class
            instance.append(thresh.astype(np.float32))

    if box!=[]:
        box      = np.array(box,np.float32)
        label    = np.array(label,np.float32)
        instance = np.array(instance,np.float32)
    else:
        box      = None
        label    = None
        instance = None

    return box, label, instance





# check ##################################################################################3
def run_check_dataset_reader():

    def augment(image, multi_mask, index):
        box, label, instance = multi_mask_to_annotation(multi_mask)

        #for display
        multi_mask = multi_mask/multi_mask.max() *255
        count  = len(instance)

        instance_gray = instance.copy()
        instance =[]
        for i in range(count):
            instance.append(
                cv2.cvtColor((instance_gray[i]*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            )
        instance = np.array(instance)
        return image, multi_mask, box, label, instance, index


    dataset = ScienceDataset(
        'train1_ids_gray_only1_500', mode='train',
        transform = augment,
    )
    sampler = SequentialSampler(dataset)
    #sampler = RandomSampler(dataset)


    for n in iter(sampler):
    #for n in range(10):
    #n=0
    #while 1:
        image, multi_mask, box, label, instance, index = dataset[n]
        image_show('image',image)
        image_show('multi_mask',multi_mask)
        count  = len(instance)
        for i in range(count):
            x0,y0,x1,y1 = box[i]
            cv2.rectangle(instance[i],(x0,y0),(x1,y1),(0,0,255),1)

            image_show('instance[i]',instance[i])
            print('label[i], box[i] : ', label[i], box[i])

            cv2.waitKey(1)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_dataset_reader()

    print( 'sucess!')

# -*- coding: utf-8 -*-
#data reader  ----------------------------------------------------------------
class XXXXXDataset(Dataset):

    def __init__(self, split, transform=None, mode='train'):
        super(XXXXXXDataset, self).__init__()
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
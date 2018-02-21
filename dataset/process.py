from common import *
from utility.file import *
from utility.draw import *

from dataset.reader import *

def multi_mask_to_overlay(multi_mask):
    overlay = skimage.color.label2rgb(multi_mask, bg_label=0, bg_color=(0, 0, 0))*255
    overlay = overlay.astype(np.uint8)
    return overlay

def thresh_to_inner_contour(thresh):
    thresh_pad = np.lib.pad(thresh, ((1, 1), (1, 1)), 'reflect')
    contour = thresh_pad[1:-1,1:-1] & (
            (thresh_pad[1:-1,1:-1] != thresh_pad[:-2,1:-1]) \
          | (thresh_pad[1:-1,1:-1] != thresh_pad[2:,1:-1])  \
          | (thresh_pad[1:-1,1:-1] != thresh_pad[1:-1,:-2]) \
          | (thresh_pad[1:-1,1:-1] != thresh_pad[1:-1,2:])
    )
    return contour



#extra processing
def run_make_annotation():

    split = 'train1_ids_all_670'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]
        image_files =   glob.glob(DATA_DIR + '/image/' + id + '/images/*.png')
        assert(len(image_files)==1)
        image_file=image_files[0]
        print(id)

        #----clear old -----------------------------
        if 1:
            for f in ['one_mask.png','one_countour_mask.png','one_countour_image.png','one_countour.png',
                      'overlap.png', 'one_center.png','/masks.npy', '/labels.npy',
                      '/countour_on_image.png', '/cut_mask.png', '/label.npy', '/mask.png','/overlay.png',
                      '/multi.npy','/multi.png',
                      '/instance.npy','/instance.png',
                      '/multi_instance.npy','/multi_instance.png',
                      ]:
                file = DATA_DIR + '/image/' + id + '/' + f
                if os.path.exists(file):
                    os.remove(file)
        #----clear old -----------------------------


        #image
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)

        H,W,C = image.shape
        multi_mask = np.zeros((H,W), np.int32)
        mask     = np.zeros((H,W), np.uint8)
        countour = np.zeros((H,W), np.uint8)




        mask_files = glob.glob(DATA_DIR + '/image/' + id + '/masks/*.png')
        mask_files.sort()
        count = len(mask_files)
        for i in range(count):
            mask_file = mask_files[i]
            thresh = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
            thresh = thresh >128
            index  = np.where(thresh==True)

            multi_mask[thresh]= i+1
            mask  = np.logical_or(mask,thresh)
            countour = np.logical_or(countour, thresh_to_inner_contour(thresh) )



        ## save and show -------------------------------------------
        countour_on_image = image.copy()
        countour_on_image = countour[:,:,np.newaxis]*np.array((0,255,0)) +  (1-countour[:,:,np.newaxis])*countour_on_image

        countour_overlay  = countour*255
        mask_overlay  = mask*255
        multi_mask_overlay = multi_mask_to_overlay(multi_mask)


        image_show('image',image)
        image_show('mask', mask_overlay)
        image_show('multi_mask',multi_mask_overlay)
        image_show('countour',countour_overlay)
        image_show('countour_on_image',countour_on_image)



        np.save(DATA_DIR + '/image/' + id + '/multi_mask.npy', multi_mask)
        cv2.imwrite(DATA_DIR + '/image/' + id + '/multi_mask.png',multi_mask_overlay)
        cv2.imwrite(DATA_DIR + '/image/' + id + '/mask.png',mask_overlay)
        cv2.imwrite(DATA_DIR + '/image/' + id + '/countour.png',countour_overlay)
        cv2.imwrite(DATA_DIR + '/image/' + id + '/countour_on_image.png',countour_on_image)

        cv2.waitKey(1)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_make_annotation()

    print( 'sucess!')

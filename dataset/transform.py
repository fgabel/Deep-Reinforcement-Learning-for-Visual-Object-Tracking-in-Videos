from common import *


## for debug
def dummy_transform(image):
    print ('\tdummy_transform')
    return image

# kaggle science bowl-2 : -------------------------------------------------------

def resize_to_factor2(image, mask, factor=16):

    H,W = image.shape[:2]
    h = (H//factor)*factor
    w = (W //factor)*factor
    return fix_resize_transform2(image, mask, w, h)



def fix_resize_transform2(image, mask, w, h):
    H,W = image.shape[:2]
    if (H,W) != (h,w):
        image = cv2.resize(image,(w,h))

        mask = mask.astype(np.float32)
        mask = cv2.resize(mask,(w,h))
        mask = mask.astype(np.int32)
    return image, mask




def fix_crop_transform2(image, mask, x,y,w,h):

    H,W = image.shape[:2]
    assert(H>=h)
    assert(W >=w)

    if (x==-1 & y==-1):
        x=(W-w)//2
        y=(H-h)//2

    if (x,y,w,h) != (0,0,W,H):
        image = image[y:y+h, x:x+w]
        mask = mask[y:y+h, x:x+w]

    return image, mask


def random_crop_transform2(image, mask, w,h):
    H,W = image.shape[:2]

    if H!=h:
        y = np.random.choice(H-h)
    else:
        y=0

    if W!=w:
        x = np.random.choice(W-w)
    else:
        x=0

    return fix_crop_transform2(image, mask, x,y,w,h)


def resize_to_factor(image, factor=16):
    height,width = image.shape[:2]
    h = (height//factor)*factor
    w = (width //factor)*factor
    return fix_resize_transform(image, w, h)


def fix_resize_transform(image, w, h):
    height,width = image.shape[:2]
    if (height,width) != (h,w):
        image = cv2.resize(image,(w,h))
    return image

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    print('\nsucess!')
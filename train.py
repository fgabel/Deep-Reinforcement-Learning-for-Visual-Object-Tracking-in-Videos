from common import *
from utils import *

# ------------------------------------------------------------------------------------
from model import DLRTnet
from utils import VOT2017Dataset

def train_augment(image, multi_mask, index):
    pass




def valid_augment(image, multi_mask, index):
    pass



def train_collate(batch):

    batch_size = len(batch)
    #for b in range(batch_size): print (batch[b][0].size())
    inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    boxes     =             [batch[b][1]for b in range(batch_size)]
    labels    =             [batch[b][2]for b in range(batch_size)]
    instances =             [batch[b][3]for b in range(batch_size)]
    indices   =             [batch[b][4]for b in range(batch_size)]

    return [inputs, boxes, labels, instances, indices]

### draw #########################################################

def draw():
    
    pass







### training ##############################################################
def evaluate(net, test_loader):

    test_num  = 0
    test_loss = np.zeros(6,np.float32)
    test_acc  = 0
    for i, (inputs, boxes, labels, instances, indices) in enumerate(test_loader, 0):
        inputs = Variable(inputs,volatile=True).cuda()

        net(inputs, boxes,  labels, instances )
        loss = net.loss(inputs, boxes,  labels, instances)

        # acc    = dice_loss(masks, labels) #todo

        batch_size = len(indices)
        test_acc  += 0 #batch_size*acc[0][0]
        test_loss += batch_size*np.array((
                           loss .cpu().data.numpy()[0],
                           net.rpn_cls_loss.cpu().data.numpy()[0],
                           net.rpn_reg_loss.cpu().data.numpy()[0],
                           net.rcnn_cls_loss.cpu().data.numpy()[0],
                           net.rcnn_reg_loss.cpu().data.numpy()[0],
                           net.mask_cls_loss.cpu().data.numpy()[0],
                         ))
        test_num  += batch_size

    assert(test_num == len(test_loader.sampler))
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num
    return test_loss, test_acc



#--------------------------------------------------------------
def run_train():

    out_dir  = RESULTS_DIR + '/mask-rcnn-gray-011a-debug'
    initial_checkpoint = \
        RESULTS_DIR + '/mask-rcnn-gray-011a-debug/checkpoint/00072200_model.pth'
        #


    pretrain_file = None #imagenet pretrain
    ## setup  -----------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/train', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## net ----------------------
    log.write('** net setting **\n')
    cfg = Configuration()
    net = MaskRcnnNet(cfg).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    elif pretrain_file is not None:
        log.write('\tpretrained_file = %s\n' % pretrain_file)
        #load_pretrain_file(net, pretrain_file)


    log.write('%s\n\n'%(type(net)))
    log.write('\n')



    ## optimiser ----------------------------------
    iter_accum  = 1
    batch_size  = 4  ##NUM_CUDA_DEVICES*512 #256//iter_accum #512 #2*288//iter_accum

    num_iters   = 1000  *1000
    iter_smooth = 20
    iter_log    = 50
    iter_valid  = 100
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,100))#1*1000


    LR = None  #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.001/iter_accum, momentum=0.9, weight_decay=0.0001)

    start_iter = 0
    start_epoch= 0.
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']
        #optimizer.load_state_dict(checkpoint['optimizer'])


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = XXXXXDataset(
                                #'train1_ids_gray_only1_500', mode='train',
                                'valid1_ids_gray_only1_43', mode='train',
                                transform = train_augment)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        #sampler = ConstantSampler(train_dataset,list(range(16))),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = train_collate)


    valid_dataset = ScienceDataset(
                                'valid1_ids_gray_only1_43', mode='train',
                                #'debug1_ids_gray_only1_10', mode='train',
                                 transform = valid_augment)
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = train_collate)

    log.write('\ttrain_dataset.split = %s\n'%(train_dataset.split))
    log.write('\tvalid_dataset.split = %s\n'%(valid_dataset.split))
    log.write('\tlen(train_dataset)  = %d\n'%(len(train_dataset)))
    log.write('\tlen(valid_dataset)  = %d\n'%(len(valid_dataset)))
    log.write('\tlen(train_loader)   = %d\n'%(len(train_loader)))
    log.write('\tlen(valid_loader)   = %d\n'%(len(valid_loader)))
    log.write('\tbatch_size  = %d\n'%(batch_size))
    log.write('\titer_accum  = %d\n'%(iter_accum))
    log.write('\tbatch_size*iter_accum  = %d\n'%(batch_size*iter_accum))
    log.write('\n')

    #log.write(inspect.getsource(train_augment)+'\n')
    #log.write(inspect.getsource(valid_augment)+'\n')
    #log.write('\n')

    if 0: #<debug>
        for inputs, truth_boxes, truth_labels, truth_instances, indices in valid_loader:

            batch_size, C,H,W = inputs.size()
            print(batch_size)

            images = inputs.cpu().numpy()
            for b in range(batch_size):
                image = (images[b].transpose((1,2,0))*255)
                image = np.clip(image.astype(np.float32)*3,0,255)

                image1 = image.copy()

                truth_box = truth_boxes[b]
                truth_label = truth_labels[b]
                truth_instance = truth_instances[b]
                if truth_box is not None:
                    for box,label,instance in zip(truth_box,truth_label,truth_instance):
                        x0,y0,x1,y1 = box.astype(np.int32)
                        cv2.rectangle(image,(x0,y0),(x1,y1),(0,0,255),1)
                        print(label)

                        thresh = instance>0.5
                        contour = thresh_to_inner_contour(thresh)
                        contour = contour.astype(np.float32) *0.5

                        image1 = contour[:,:,np.newaxis]*np.array((0,255,0)) +  (1-contour[:,:,np.newaxis])*image1


                    print('')


                image_show('image',image)
                image_show('image1',image1)
                cv2.waitKey(0)



    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' momentum=%f\n'% optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n'%str(LR) )

    log.write(' images_per_epoch = %d\n\n'%len(train_dataset))
    log.write(' rate    iter   epoch  num   | valid_loss                           | train_loss                           | batch_loss                           |  time    \n')
    log.write('------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss  = np.zeros(6,np.float32)
    train_acc   = 0.0
    valid_loss  = np.zeros(6,np.float32)
    valid_acc   = 0.0
    batch_loss  = np.zeros(6,np.float32)
    batch_acc   = 0.0
    rate = 0

    start = timer()
    j = 0
    i = 0


    for i in range(n_epochs):  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6, np.float32)
        sum_train_acc  = 0.0
        sum = 0

        net.set_mode('train')
        optimizer.zero_grad()
        for inputs, truth_boxes, truth_labels, truth_instances, indices in train_loader:
            batch_size = len(indices)
            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*iter_accum/len(train_dataset) + start_epoch
            num_products = epoch*len(train_dataset)

            if i % iter_valid==0:
                net.set_mode('valid')
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.set_mode('train')

                print('\r',end='',flush=True)
                log.write('%0.4f %5.1f k %6.2f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s\n' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5], #valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5], #train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5], #batch_acc,
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            #if 1:
            if i in iter_save:
                torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : i,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(i))



            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)[0]*iter_accum




            # one iteration update  -------------
            inputs = Variable(inputs).cuda()
            net( inputs, truth_boxes, truth_labels, truth_instances )
            loss = net.loss( inputs, truth_boxes, truth_labels, truth_instances )


            if 1: #<debug>
                debug_and_draw(net, inputs, truth_boxes, truth_labels, truth_instances, mode='test')

            # masks  = (probs>0.5).float()
            # acc    = dice_loss(masks, labels)


            # accumulated update
            loss.backward()
            if j%iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()


            # print statistics  ------------
            batch_acc  = 0 #acc[0][0]
            batch_loss = np.array((
                           loss.cpu().data.numpy()[0],
                           net.rpn_cls_loss.cpu().data.numpy()[0],
                           net.rpn_reg_loss.cpu().data.numpy()[0],
                           net.rcnn_cls_loss.cpu().data.numpy()[0],
                           net.rcnn_reg_loss.cpu().data.numpy()[0],
                           net.mask_cls_loss.cpu().data.numpy()[0],
                         ))
            sum_train_loss += batch_loss
            sum_train_acc  += batch_acc
            sum += 1
            if i%iter_smooth == 0:
                train_loss = sum_train_loss/sum
                train_acc  = sum_train_acc /sum
                sum_train_loss = np.zeros(6,np.float32)
                sum_train_acc  = 0.
                sum = 0


            print('\r%0.4f %5.1f k %6.2f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s  %d,%d,%s' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5], #valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5], #train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5], #batch_acc,
                         time_to_str((timer() - start)/60) ,i,j, str(inputs.size())), end='',flush=True)
            j=j+1



        pass  #-- end of one data loader --
    pass #-- end of all iterations --


    if 1: #save last
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')

def train(model, criterion, optimizer, n_epochs, T):
    # Train the Model
    vot_data = VOT2017_dataset(csv_file= 'F:/vot2017/list.txt',
                       root_dir= 'F:/vot2017/')  
    train_loader = dataloader(vot_data) # iterating over this gives videos
    for vid in train_loader:
        for image in vid[...]:
            
    for epoch in range(n_epochs):
        
        for i, video in enumerate(train_loader):
            # video is now a dict of video, gt pairs
            image_stack = video['video'] # shape(Nr. of images, h, w, RGB)
            masks = video['gt']
            current_pos_of_t = 0 # start from the start of the video
            for t in range(T):
                """we take a T-image sequence out of the video"""     
                if current_pos_of_t + T > image_stack.shape[0]:
                    """This loop makes sure that we dont get an index error. Handling the edges like this is not so nice though"""
                    continue
                image_stack_temp = image_stack[current_pos_of_t: current_pos_of_t + T, :, :, :]
                masks_temp = masks[current_pos_of_t: current_pos_of_t + T, :, :] # check FORMATS!
                current_pos_of_t += T # next iteration, take the next T-image sequence
                reward_list = []
                b_t_list = []
                for image, mask in zip(image_stack_temp, masks_temp): 
                    """ iterate over the first dimensionof image_stack_temp, the number of images
                        image is now a particular image of the sequence, mask its corresonding mask
                    """
                    image = Variable(mask.view(-1, sequence_length, input_size))
                    mask = Variable(mask)
                
                    # Forward + Backward + Optimize
                    optimizer.zero_grad() # reset gradients
                    outputs = DLRTnet(images) # observation network
                    out, hidden = LSTM(outputs, hidden) # LSTM gets the output of DLRTnet as input and its previous hidden state
                    """GaussianLayer takes a hidden layer and samples N masks from the last 4 / 8 numbers"""
                    l_t = GaussianLayer(hidden) # l_t contains N sampled masks
                    
                    for mask_ in l_t:
                        reward += LOSSFUNCTION(l_t, mask) # take the loss function 1 from the paper and 
                    reward_list.append(reward) # this list contains r1, r2, r3, ..., rT
                    b_t_list.append(1/N*reward) # this contains b1, b2, ... bT
                
                
                # Compute gradient
                image_reward = np.asarray(reward_list)
                image_base = np.asarray(b_t_list)
                    
                    
                loss.backward() # Crucial step here is to implement backward pass of GaussianLayer using reward_list
                optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                           %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()

    print('\nsucess!')



#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#
#
from common import *


def weighted_binary_cross_entropy_with_logits(logits, labels, weights):

    loss = logits.clamp(min=0) - logits*labels + torch.log(1 + torch.exp(-logits.abs())) 
    loss = (weights*loss).sum()/(weights.sum()+1e-12)

    return loss
	
	
def binary_cross_entropy_with_logits(logits, labels):

    loss = logits.clamp(min=0) - logits*labels + torch.log(1 + torch.exp(-logits.abs()))
    loss = loss.sum()/len(loss)

    return loss


def mask_loss(logits, labels, instances ): 
 
    batch_size, num_classes = logits.size(0), logits.size(1)

    logits_flat = logits.view (batch_size,num_classes, -1)
    dim =  logits_flat.size(2)

    # one hot encode
    select = Variable(torch.zeros((batch_size,num_classes))).cuda()
    select.scatter_(1, labels.view(-1,1), 1)
    select[:,0] = 0
    select = select.view(batch_size,num_classes,1).expand((batch_size,num_classes,dim)).contiguous().byte()

    logits_flat = logits_flat[select].view(-1)
    labels_flat = instances.view(-1)
 
    loss = binary_cross_entropy_with_logits(logits_flat, labels_flat)
    return loss



# #-----------------------------------------------------------------------------
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

 
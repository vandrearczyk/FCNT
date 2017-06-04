import caffe
import surgery, score
import numpy as np
import os, errno
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import net_fcnT


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError, e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def bigBlobs(img, nb_class):
    "Returns an image with the biggest blobs set to their class label and small blobs set to -1"
    bigblobs = np.full(img.shape, 255, np.uint8)
    areas = np.zeros(nb_class)
    
    # Sort from largest to smallest blob
    for c in range(nb_class):
        im_th = np.copy(img)
        im_th[img==c]=255
        im_th[img!=c]=0
        # Extract the biggest blobs
        im2, contours, hierarchy = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt)>areas[c]:
                areas[c] = cv2.contourArea(cnt)
                
    for c in np.argsort(-areas):
        # Threshold the img image for the img class label
        im_th = np.copy(img)
        im_th[img==c]=255
        im_th[img!=c]=0
        # create a mask
        mask = np.zeros(im_th.shape,np.uint8)
        # Extract the biggest blobs 
        im2, contours, hierarchy = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        area=0
        biggest=0
        i_cnt=0
        for cnt in contours:
            if cv2.contourArea(cnt)>area:
                biggest = i_cnt
                area = cv2.contourArea(cnt)
            i_cnt+=1
        cv2.drawContours(mask,contours,biggest,255,-1)
        # Workaround for the edges problem: pad with "edge"
        mask = np.pad(mask[1:-1,1:-1], pad_width=1, mode='edge')
        # Set the values of bigblobs
        bigblobs[mask==255] = c
        
    return bigblobs

def relabelSmallBlobs(img, best):
    "Returns an image with the small isolated blobs (255 in the input image) set to the values in best (Xth best prediction from the CNN output.)"
    relabel = img.copy()
    relabel[img==255] = best[img==255]
    return relabel

def main(args):
    data = str(args.data)+'_1' # Images (and labels) saved as 1_1.jpg, 2_1.jpg, 3_1.jpg etc. in root-fcn/data/prague_normal
    n_iter = args.n_iter
    
    ###### Train ######
    print 'start train'
    
    # Link the correct train.txt and val.txt
    symlink_force('train'+data+'.txt', '../data/prague_normal/ImageSets/Segmentation/train.txt')
    symlink_force('val'+data+'.txt' , '../data/prague_normal/ImageSets/Segmentation/val.txt')
    
    # Link the ground truth for training
    i=0
    with open('../data/prague_normal/ImageSets/Segmentation/train.txt', 'r') as f:
        trainlist = f.read().splitlines() 
        for line in trainlist:
            if 'grid4_t0' not in line: # grid4 is used only in set 14 and has a different shape. We keep it linked to the same padded image.
                symlink_force('gt'+str(i)+'.png', '../data/prague_normal/SegmentationClass/'+line+'.png')
            else:
                symlink_force('gt_grid4_'+str(i)+'.png', '../data/prague_normal/SegmentationClass/'+line+'.png')
            i+=1
    
    # Import ground truth
    true_label = Image.open('../data/prague_normal/SegmentationClass/'+data+'.png')
    nb_class = len(np.unique(true_label)) # also given by the number of train images but it's easier to obtin it like this
    true_label = np.array(true_label, dtype=np.float32)
        
    # Create train.prototxt and val.prototxt
    net_fcnT.make_net(nb_class)
    
    # Load weights
    weights = '../voc-fcn8s/fcn8s-heavy-pascal.caffemodel'
    base_net = caffe.Net('../voc-fcn8s/deploy.prototxt', weights, caffe.TEST)
    
    
    # init
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver('solver.prototxt')
    solver.net.copy_from(weights)
    # surgeries
    surgery.transplant(solver.net,base_net)
    
    # Train for n_iter iterations and save the model
    solver.step(n_iter)
    if not os.path.exists('snapshot'):
        os.makedirs('snapshot')
    solver.net.save('snapshot/prague.caffemodel')
    
    # Copy the train.prototxt to deploy.prototxt (for testing)
    f = open('train.prototxt')
    l_train = f.readlines()
    f.close()
    f = open('deploy.prototxt','r')
    l_deploy = f.readlines()
    f.close()
    with open('deploy.prototxt','w') as f:
        for _, line in enumerate(l_deploy[:10]):
            f.write(line)
        for _, line in enumerate(l_train[11:-11]):
            f.write(line)
            
    ###### Test ######
    # Import and prepare test image
    im = Image.open('../data/prague_normal/JPEGImages/'+data+'.jpg')
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    
    # load trained net
    net = caffe.Net('deploy.prototxt', 'snapshot/prague.caffemodel', caffe.TEST)
    
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    outnopost = net.blobs['score'].data[0].argmax(axis=0)
    
    ###### Refinement method ######
    current = outnopost.copy()
    past = np.zeros(current.shape,np.uint8)
    nb_clust = len(np.unique(current))
    while 1:
        for n in range(nb_clust):
            past = current.copy()
            current = relabelSmallBlobs(current,np.argsort(-net.blobs['score'].data[0], axis=0)[n])
            current = bigBlobs(current.astype(np.uint8), nb_clust)
            if not np.array_equal(current, past):
                break
        if 255 not in current: # 255 represents thesmall isolated regnions not part of the biggest blobs.
            break
    out= current.copy()
    # Measure correct pixel assignment (CO)
    w, h = out.shape
    accu = round(float(np.sum(true_label==out))/float(w*h),4)
    
    # Plot and save
    fig = plt.figure(figsize = (20,20))
    fig.add_subplot(2,2,1)
    imgplot = plt.imshow(im)
    plt.axis('off')
    plt.title('input image')
    fig.add_subplot(2,2,2)
    imgplot = plt.imshow(true_label)
    plt.axis('off')
    plt.title('ground truth segmentation')
    fig.add_subplot(2,2,3)
    imgplot = plt.imshow(outnopost)
    plt.axis('off')
    plt.title('segmentation before refinement')
    fig.add_subplot(2,2,4)
    imgplot = plt.imshow(out)
    plt.axis('off')
    plt.title('segmentation after refinement (CO='+str(accu)+')') 
    
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/out_'+data+'.jpg', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised texture segmentation of the Prague normal dataset.')
    parser.add_argument('--data', metavar='data', type=int, default=1,
                        help='number of the test image in the Prague normal texture segmentation dataset (1..20). Default 1')
    parser.add_argument('--n_iter', metavar='n_iter', type=int, default=500,
                        help='number of training iterations. Default 500')
    
    main(parser.parse_args())
import sys
sys.path.append('../../python')

import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='xavier'))
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fcn(split,nb_class):
    n = caffe.NetSpec()
    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
            seed=1337)
    pydata_params['prague_dir'] = '../data/prague_normal'
    pylayer = 'PRAGUESegDataLayer'
    n.data, n.label = L.Python(module='prague_layers', layer=pylayer,
            ntop=2, param_str=str(pydata_params))

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)
    
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)
    
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)
    
    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)
    
    # fully conv
    n.fc6b, n.relu6 = conv_relu(n.pool4, 4096, ks=5, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7b, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    n.score_frb = L.Convolution(n.drop7, num_output=nb_class, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    
    n.upscore1b = L.Deconvolution(n.score_frb,
        convolution_param=dict(num_output=nb_class, group=nb_class, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=0,decay_mult=0)],
        weight_filler=dict(type='bilinear'))
    
    n.score_pool3b = L.Convolution(n.pool3, num_output=nb_class, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.score_pool3c = crop(n.score_pool3b, n.upscore1b)
    n.fuse_pool3 = L.Eltwise(n.upscore1b, n.score_pool3c,
            operation=P.Eltwise.SUM)
    
    n.upscore2b = L.Deconvolution(n.fuse_pool3,
        convolution_param=dict(num_output=nb_class, group=nb_class, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=0,decay_mult=0)],
        weight_filler=dict(type='bilinear'))    
    
    n.score_pool2b = L.Convolution(n.pool2, num_output=nb_class, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.score_pool2c = crop(n.score_pool2b, n.upscore2b)
    n.fuse_pool2 = L.Eltwise(n.upscore2b, n.score_pool2c,
            operation=P.Eltwise.SUM)
    
    n.upscore3b = L.Deconvolution(n.fuse_pool2,
        convolution_param=dict(num_output=nb_class, group=nb_class, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=0,decay_mult=0)],
        weight_filler=dict(type='bilinear'))

    n.score_pool1b = L.Convolution(n.pool1, num_output=nb_class, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.score_pool1c = crop(n.score_pool1b, n.upscore3b)
    n.fuse_pool1 = L.Eltwise(n.upscore3b, n.score_pool1c,
            operation=P.Eltwise.SUM)
    
    n.upscore4b = L.Deconvolution(n.fuse_pool1,
        convolution_param=dict(num_output=nb_class, group=nb_class, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=0,decay_mult=0)],
        weight_filler=dict(type='bilinear'))   
    
    n.score = crop(n.upscore4b, n.data)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=255))

    return n.to_proto()

def make_net(nb_class):
    with open('train.prototxt', 'w') as f:
        f.write(str(fcn('train',nb_class)))

    with open('val.prototxt', 'w') as f:
        f.write(str(fcn('val',nb_class)))

if __name__ == '__main__':
    make_net(int(sys.argv[1]))

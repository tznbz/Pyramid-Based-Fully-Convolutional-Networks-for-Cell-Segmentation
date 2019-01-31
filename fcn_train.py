##-------------------------------------------------------------------
## Copyright (C) 201 Tianyi Zhao
## File : fcn_train.py
## Author : Tianyi Zhao <t.zhao0321@gmail.com>
## Description :
## --
## Created : <2017-05-25>
## Updated: Time-stamp: <2017-05-25 22:49:10>
##-------------------------------------------------------------------


import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import tensorflow as tf
import time
from scipy.ndimage import zoom
from skimage import io
import nibabel as nib
from skimage.measure import label, regionprops
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import erosion,dilation,square
from skimage.filters import gaussian
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement = True


batchsize = 4

def weight_variable(shape,stddev=0.05):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W,st=1,pad=0):
    return tf.nn.conv2d(x, W, strides=[1, st, st, 1], padding='SAME')

def conv2d_BN(x,W,st=1,pad=0):
    epsilon = 1e-3
    h_conv = tf.nn.conv2d(x, W, strides=[1, st, st, 1], padding='SAME')
    batch_mean2, batch_var2 = tf.nn.moments(h_conv,[0,1,2])
    scale2 = tf.Variable(tf.ones([W.shape[-1]]))
    beta2 = tf.Variable(tf.zeros([W.shape[-1]]))
    h_fl3_BN2 = tf.nn.batch_normalization(h_conv,batch_mean2,batch_var2,beta2,scale2,epsilon)
    return h_fl3_BN2



def createweights():
    W_conv1 = weight_variable([3, 3, 1, 64])
    b_conv1 = bias_variable([64])
    W_conv2 = weight_variable([3, 3, 64,64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    W_conv4 = weight_variable([3, 3, 128, 128])
    b_conv4 = bias_variable([128])
    
    W_conv5 = weight_variable([3, 3, 128, 256])
    b_conv5 = bias_variable([256])
    W_conv6 = weight_variable([3, 3, 256, 256],stddev=0.01)
    b_conv6 = bias_variable([256])

    W_conv7 = weight_variable([3, 3, 256, 128],stddev=0.01)
    b_conv7 = bias_variable([128])
    W_conv8 = weight_variable([3, 3, 128, 128],stddev=0.01)
    b_conv8 = bias_variable([128])


    W_conv9 =  weight_variable([2, 2, 128, 64],stddev=0.01)
    b_conv9 =  bias_variable([64])
    W_conv10 = weight_variable([3, 3, 64, 64],stddev=0.001)
    b_conv10 = bias_variable([64])

    W_conv11 =  weight_variable([1, 1, 64, 2])
    b_conv11 =  bias_variable([2])



    weights = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,W_conv6,W_conv7,W_conv8,W_conv9,W_conv10,
    W_conv11]
    beights = [b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,b_conv6,b_conv7,b_conv8,b_conv9,b_conv10,
    b_conv11]

    
    return weights,beights


def ConvNetwork(x_image,weights,beights):
    #x = tf.placeholder(tf.float64, shape=[None, 256,256,1])
    #xm = tf.placeholder(tf.float64, shape=[None, 32,32,1]) 

    h_conv1 = tf.nn.relu(conv2d_BN(x_image, weights[0]) + beights[0])
    h_conv2 = tf.nn.relu(conv2d_BN(h_conv1, weights[1]) + beights[1])

    h_conv3 = tf.nn.relu(conv2d_BN(h_conv2, weights[2]) + beights[2])
    h_conv4 = tf.nn.relu(conv2d_BN(h_conv3, weights[3]) + beights[3])

    h_conv5 = tf.nn.relu(conv2d_BN(h_conv4, weights[4]) + beights[4])
    h_conv6 = tf.nn.relu(conv2d_BN(h_conv5, weights[5]) + beights[5])


    h_conv7 = tf.nn.relu(conv2d_BN(h_conv6, weights[6]) + beights[6])
    h_conv8 = tf.nn.relu(conv2d_BN(h_conv7, weights[7]) + beights[7])

    h_conv9 = tf.nn.relu(conv2d(h_conv8, weights[8]) + beights[8])
    h_conv10 = tf.nn.relu(conv2d(h_conv9, weights[9]) + beights[9])



    h_conv11 = (conv2d(h_conv10, weights[10]) + beights[10])
    
  

    
    #h_conv23 = (conv2d(h_conv22, weights[22]) + beights[22])


    softmax = tf.nn.softmax(h_conv11)


    return h_conv11,softmax

    
def dice_coieffience_loss(result_mask,x_mask):
    eps = 1e-5
    intersection =  tf.reduce_sum(result_mask * x_mask,axis=[1,2,3])
    union =  eps + tf.reduce_sum(result_mask,axis=[1,2,3]) + tf.reduce_sum(x_mask,axis=[1,2,3])
    loss = tf.reduce_mean(-2 * intersection/ (union))
    #
    return loss

def cross_entropy_loss(result_mask,x_mask,scale=512):
    loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.reshape(result_mask,[batchsize*scale*scale,2]),  labels=tf.reshape(x_mask,[batchsize*scale*scale]))
    print('lossssss',loss1)
    loss = tf.reduce_sum(loss1)
    return loss/batchsize/1000

def log_loss(result_mask,x_mask,scale=512):
    loss1 = tf.losses.log_loss(
            predictions=tf.reshape(result_mask,[batchsize*scale*scale,2]),  labels=tf.reshape(x_mask,[batchsize*scale*scale,2]))
    #print('lossssss',loss1)
    loss = tf.reduce_sum(loss1)
    return loss
    
    
def run_training(threshold = 0.47,kfolderid = 1,loadmodel = False):
    with tf.Graph().as_default():
        #data
        print('loading data')
        if True:
            ###01
            trainiff_dir = './PhC-C2DH-U373/01/'  
            labeliff_dir = './PhC-C2DH-U373/01_GT/SEG/' 
            tif_list = os.listdir(labeliff_dir)
            tif_list.sort()
            dataf_list = [trainiff_dir+'t'+f[-7:] for f in tif_list]
            labelf_list = [labeliff_dir+f for f in tif_list]
            ###02
            trainiff_dir = './PhC-C2DH-U373/02/'  
            labeliff_dir = './PhC-C2DH-U373/02_GT/SEG/' 
            tif_list = os.listdir(labeliff_dir)
            tif_list.sort()
            dataf_list += [trainiff_dir+'t'+f[-7:] for f in tif_list]
            labelf_list += [labeliff_dir+f for f in tif_list]

            im = np.expand_dims(io.imread(dataf_list[0]).astype(np.float32),axis=-1)
            lbs = np.expand_dims(io.imread(labelf_list[0]).astype(np.float32),axis=-1)
            lbs[lbs>0] = 1
            im = np.expand_dims(im,axis=0)
            lbs = np.expand_dims(lbs,axis=0)
            for jj in range(1,len(dataf_list)):
                imi = np.expand_dims(io.imread(dataf_list[jj]).astype(np.float32),axis=-1)
                lbsi = np.expand_dims(io.imread(labelf_list[jj]).astype(np.float32),axis=-1)
                lbsi[lbsi>0] = 1
                imi = np.expand_dims(imi,axis=0)
                lbsi = np.expand_dims(lbsi,axis=0)
                im = np.concatenate([im,imi])
                lbs = np.concatenate([lbs,lbsi])

            ###resize
            im = zoom(im,(1,512./im.shape[1],512./im.shape[2],1),order=1 )
            lbs = zoom(lbs,(1,512./lbs.shape[1],512./lbs.shape[2],1),order=1 )


        #tf define the input
        images512 = tf.placeholder(tf.float32, [batchsize, 512,512,1])
        images256 = tf.image.resize_images(images512,[256,256])
        images128 = tf.image.resize_images(images256,[128,128])
        #tf.image.resize_images
        masks512 = tf.placeholder(tf.float32, [batchsize, 512,512,1])
        masks256 = tf.image.resize_images(masks512,[256,256])
        masks128 = tf.image.resize_images(masks256,[128,128])
        imasks128 = tf.cast(masks128, tf.int32) 
        imasks256 = tf.cast(masks256, tf.int32) 
        imasks512 = tf.cast(masks512, tf.int32) 

        #Graph
        print('building graph')
        weights128,beights128 = createweights()
        weights256,beights256 = createweights()
        weights512,beights512 = createweights()
        h5_128,h5_softmax128 = ConvNetwork(images128,weights128,beights128)
        h5_256,h5_softmax256 = ConvNetwork(images256,weights256,beights256)
        h5_512,h5_softmax512 = ConvNetwork(images512,weights512,beights128)
        print('output is',h5_512)

        upop = tf.contrib.keras.layers.UpSampling2D((2,2),'channels_last')
        pre_mask1 = tf.expand_dims((h5_softmax128)[:,:,:,1],-1) 
        dcloss1 = dice_coieffience_loss(pre_mask1,masks128)
        pre_mask2 = tf.expand_dims(((h5_softmax256+upop(h5_softmax128))/2)[:,:,:,1],-1) 
        dcloss2 = dice_coieffience_loss(pre_mask2,masks256)
        pre_mask3 = tf.expand_dims(((h5_softmax512+upop(h5_softmax256))/2)[:,:,:,1],-1) 
        dcloss3 = dice_coieffience_loss(pre_mask3,masks512)
    
        loss128 = cross_entropy_loss(h5_128,imasks128,128)
        loss256 = log_loss((h5_softmax256+upop(h5_softmax128))/2,tf.concat([1-masks256,masks256],3) ,256)
        loss512 = log_loss((h5_softmax512+upop(h5_softmax256))/2,tf.concat([1-masks512,masks512],3) ,512)

        #Train
        lr=0.0001
        train_step128 = tf.train.MomentumOptimizer(lr,0.9).minimize(loss128, var_list=weights128+beights128)
        train_step256 = tf.train.MomentumOptimizer(lr,0.9).minimize(loss256, var_list=weights256+beights256)
        train_step512 = tf.train.MomentumOptimizer(lr,0.9).minimize(loss512, var_list=weights512+beights512)
        #Run
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        

        
        
        saver256 = tf.train.Saver({'Variable':weights256[0],
            'Variable_1':beights256[0],
            'Variable_2':weights256[1],
            'Variable_3':beights256[1],
            'Variable_4':weights256[2],
            'Variable_5':beights256[2],
            'Variable_6':weights256[3],
            'Variable_7':beights256[3],
            'Variable_8':weights256[4],
            'Variable_9':beights256[4],
            'Variable_10':weights256[5],
            'Variable_11':beights256[5],
            'Variable_12':weights256[6],
            'Variable_13':beights256[6],
            'Variable_14':weights256[7],
            'Variable_15':beights256[7],
            'Variable_16':weights256[8],
            'Variable_17':beights256[8],
            'Variable_18':weights256[9],
            'Variable_19':beights256[9],
            'Variable_20':weights256[10],
            'Variable_21':beights256[10]
                })
        

        
        
        saver3 = tf.train.Saver(max_to_keep=1000)
        sess =   tf.Session(config=config)  
        print('run initial')
        sess.run(init_op)
        print('coordinator')


        #datawet others
        expdir = './experiment_fcn/'
        if not os.path.exists(expdir):
            os.makedirs(expdir)
        elif loadmodel:
            flist = os.listdir(expdir)
            if flist:
                flist = [expdir + f for f in flist]
                flist.sort(key=os.path.getmtime )
                rstdir = flist[-1] + '/model.ckpt' 
                saver3.restore(sess,rstdir)
                print('Model retored from',rstdir)

        
        #saver256.restore(sess,rstdir)
        #print('Model retored from',rstdir)      

        kk=0
        loss_all = np.zeros(6)
        print_freqence = 20
        save_freqence = 200
        kk = 0
        pointer = 0
        shuffleidx = np.arange(len(im))
        np.random.shuffle(shuffleidx)
        
        try:
            while True:
                #load data
                point_end = pointer+batchsize
                if point_end>len(im):
                    batch_xs = np.concatenate( [im[shuffleidx[pointer:]],im[shuffleidx[:batchsize-(len(im)-pointer)]]], axis=0)
                    batch_ys = np.concatenate( [lbs[shuffleidx[pointer:]],lbs[shuffleidx[:batchsize-(len(im)-pointer)]]], axis=0)
                    pointer=batchsize-(len(im)-pointer)
                else:
                    batch_xs = im[shuffleidx[pointer:point_end]]
                    batch_ys = lbs[shuffleidx[pointer:point_end]]
                    pointer+=batchsize

                #training
                if kk <10000:
                    [_,dd,dd2,dd3,ll,l2,l3] = sess.run([train_step128,dcloss1,dcloss2,dcloss3,loss128,loss256,loss512],feed_dict={images512: batch_xs, masks512: batch_ys})
                elif kk <20000:
                    [_,dd,dd2,dd3,ll,l2,l3] = sess.run([train_step256,dcloss1,dcloss2,dcloss3,loss128,loss256,loss512],feed_dict={images512: batch_xs, masks512: batch_ys})
                elif kk <30000:
                    [_,dd,dd2,dd3,ll,l2,l3] = sess.run([train_step512,dcloss1,dcloss2,dcloss3,loss128,loss256,loss512],feed_dict={images512: batch_xs, masks512: batch_ys})
                else:
                    break
                loss_all += [dd,dd2,dd3,ll,l2,l3]
               

                #saving
                if kk%save_freqence==0:
                    if kk==0:
                        dirname =expdir+'k'+str(kfolderid)+'_DC'+str(loss_all[-2])[1:6]+'_ep'+str(kk)
                    else:
                        dirname =expdir+'k'+str(kfolderid)+'_DC'+str(loss_all[-2]/print_freqence)[1:6]+'_ep'+str(kk)
                   
                    while os.path.exists(dirname):
                        dirname = dirname+'_'
                    os.makedirs(dirname)
                    save_path = saver3.save(sess, dirname+"/model.ckpt")
                    print("Model saved in file: %s" % save_path)
            
            
                if kk%print_freqence==0:
                    if kk==0:
                        print(kk,time.strftime("%H:%M:%S"), 'dcloss x3,loss x3',loss_all,'lr',lr)
                    else:
                        print(kk,time.strftime("%H:%M:%S"), 'dcloss x3,loss x3',loss_all/print_freqence,'lr',lr)
                    loss_all[:] = 0
         
                kk += 1
                


        except (tf.errors.OutOfRangeError,KeyboardInterrupt) as e:
            print('Done testing for %d steps.' % (kk))


        sess.close()



if __name__ == '__main__':
    run_training()




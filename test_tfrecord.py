#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:53:54 2017

@author: optimal
"""

from scipy import misc
import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

root_path = '/mount/temp/WZG/Multitask/Data/'
tfrecord_filename = root_path + 'tfrecords/test.tfrecords'

def read_and_decode(filename_queue, random_crop=False, random_clip=False, shuffle_batch=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'name': tf.FixedLenFeature([], tf.string),                           
          'image_raw': tf.FixedLenFeature([], tf.string),
          'mask_raw': tf.FixedLenFeature([], tf.string),                               
          'label': tf.FixedLenFeature([], tf.int64)
      })
    
    image = tf.decode_raw(features['image_raw'], tf.float64)
    image = tf.reshape(image, [300,300,3])
    
    mask = tf.decode_raw(features['mask_raw'], tf.float64)
    mask = tf.reshape(mask, [300,300])
    
    name = features['name']
    
    label = features['label']
    width = features['width']
    height = features['height']
    
#    if random_crop:
#        image = tf.random_crop(image, [227, 227, 3])
#    else:
#        image = tf.image.resize_image_with_crop_or_pad(image, 227, 227)
      
#    if random_clip:
#        image = tf.image.random_flip_left_right(image)
      
    
    if shuffle_batch:
        images, masks, names, labels, widths, heights = tf.train.shuffle_batch([image, mask, name, label, width, height],
                                                batch_size=4,
                                                capacity=8000,
                                                num_threads=4,
                                                min_after_dequeue=2000)
    else:
        images, masks, names, labels, widths, heights = tf.train.batch([image, mask, name, label, width, height],
                                        batch_size=4,
                                        capacity=8000,
                                        num_threads=4)
    return images, masks, names, labels, widths, heights
    

  
########################## Test ######################################
def test_run(tfrecord_filename):
    filename_queue = tf.train.string_input_producer([tfrecord_filename],
                                                    num_epochs=3)
    images, masks, names, labels, widths, heights = read_and_decode(filename_queue)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    meanfile = sio.loadmat(root_path + 'mats/mean300.mat')
    meanvalue = meanfile['mean']


    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for i in range(1):
            imgs, msks, nms, labs, wids, heis = sess.run([images, masks, names, labels, widths, heights])
            print 'batch' + str(i) + ': '
            #print type(imgs[0])

            for j in range(4):
                print nms[j] + ': ' + str(labs[j]) + ' ' + str(wids[j]) + ' ' + str(heis[j])
                img = np.uint8(imgs[j] + meanvalue)
                msk = np.uint8(msks[j])
                plt.subplot(4,2,j*2+1)
                plt.imshow(img)
                plt.subplot(4,2,j*2+2)
                plt.imshow(msk, vmin=0, vmax=5)
            plt.show()
        
        coord.request_stop()
        coord.join(threads)
        
############################ Main Function #############################
if __name__ == '__main__':
    test_run(tfrecord_filename)
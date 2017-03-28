# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 00:30:09 2017

@author: Shao Yn
"""

import tensorflow as tf
import numpy as np
import scipy.io

vgg_layers = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)


def read_vgg():
    vgg = scipy.io.loadmat()
    mean_value = vgg['normalization'][0][0][0]
    mean = np.mean(mean_value, axis=(0,1))
    weights = vgg['layers'][0]
    return weights, mean

def compute_value(input_image, weights):
    value = {}
    current = input_image
    for i, name in enumerate(vgg_layers):
        category = name[:4]
        if category == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = np.transpose(kernels, (1,0,2,3))
            #kernals.shape
            #bias.shape
            #weights.shape
            bias = bias.reshape(-1)
            #???
            conv = tf.nn.conv2d(current, tf.constant(weights), strides=(1,1,1,1), padding = 'SAME')    
            current = tf.nn.bias_add(conv, bias)
        elif category == 'relu':
            current = tf.nn.relu(current)
        elif category == 'pool':
            current = tf.nn.avg_pool(current, ksize=(1,2,2,1), strides = (1,2,2,1),
                                     padding = 'SAME')
        #记录各层的输出值
        value[name] = current
    return value
            
def preprocess(image, mean_pixel):
    return image - mean_pixel
    
    
def unprocess(image, mean_pixel):
    return image + mean_pixel
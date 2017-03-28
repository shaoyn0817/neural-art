# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 01:23:52 2017

@author: Shao Yn
"""

import vggnet

import tensorflow as tf
import numpy as np

from sys import stderr

from PIL import Image

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

def convert(content_image, style_image):
    content_shape = content_image.shape + (1,)
    style_shape = style_image.shape + (1,)
    content_features = {}
    style_features = {}
    vgg_weight, vgg_mean = vggnet.read_vgg()
    
    layer_weight = 1.0
    style_layer_weight = {}
    
    for style_layer in STYLE_LAYERS:
        ##????
        style_layer_weight[style_layer] = layer_weight
        layer_weight *= 1
        
    layer_sum = 0
    ##why need normalization
    for style_layer in STYLE_LAYERS:
        layer_sum += style_layer_weight[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layer_weight[style_layer] /= layer_sum


    g = tf.Graph()
    with g.as_default, g.device('/gpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape = content_shape)
        net = vggnet.computevalue(image, vgg_weight)
        #???
        content_pre = np.array([vggnet.preprocess(content_image, vgg_mean)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image:content_pre})
        
    g = tf.Graph()
    with g.as_default, g.device('/gpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape = style_shape)
        net = vggnet.computevalue(image, vgg_weight)
        style_pre = np.array([vggnet.preprocess(style_image, vgg_mean)])
        for layer in STYLE_LAYERS:
            style_features = net[layer].eval(feed_dict={style_pre})
            #?????
            style_features = np.reshape(style_features, (-1,style_features.shape[3]))
            gram = np.mutual(style_features.T, style_features/ features.size)
            style_features[layer] = gram

    ## check  可能为0
    initial_content_noise_coeff = 0;
    
    
    with tf.Graph().as_default():
        ##check 
        noise = np.random.normal(size=content_shape, scale=np.std(content_image) * 0.1)
        initial = tf.random_normal(content_shape)*0.256
    
        image = tf.Variable(initial)
        net = vggnet.compute_value(content_image, vgg_weight)
        
        content_layer_weight = {}
        content_layer_weight['relu4_2'] = 1
        content_layer_weight['relu5_2'] = 0;

        content_loss = 0;
        content_lossall = []
        #content_loss
        for content_layer in CONTENT_LAYERS:
            content_lossall.append(content_layer_weight[content_layer]*5e0*(2*tf.nn.l2_loss(
                                   net[content_layer]-content_features[content_layer]) /
                                     content_features[content_layer].size ))
        content_loss += reduce(tf.add, content_lossall)
        
        #style loss
        style_loss = 0
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            #????????
            _, height, width, number = layer.get_shape()
            size = height * width * number
            feats = tf.reshape(layer,(-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) /size
            style_gram = style_features[style_layer]
            style_losses.append(style_layer_weight[style_layer]*2*tf.nn.l2_loss(gram-style_gram)/style_gram.size)
        style_loss += 5e2*1*reduce(tf.add,style_losses)
        
        
        #check paper !!!!!!!
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = 1e2 * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:content_shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:content_shape[2]-1,:]) /
                    tv_x_size))
        
        loss = content_loss + style_loss + tv_loss
        
        train_step = tf.train.AdamOptimizer(1e1, 0.9, 0.999, 1e-08).minimize(loss)
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print 'Optimization started...\n'
            for i in range(1000):
                last_step = (i == 999)
                print i+" th Iteration running"
                train_step.run()
                if i == 999:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()
                    
                    img_out = vggnet.unprocess(best.reshape(shape[1:]),vgg_mean)
                    yield (
                        (None if last_step else i),
                        img_out
                    )
                        
                        
def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1) 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:16:52 2018

@author: fredman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:12:14 2018

@author: fredman
"""

########################################################################
###
###  This file creates takes in input images, and alignes them, rturning the
###  alined images, and the alignment data loss and regularization loss.
###  The alignment is achieved by using a pixel-location variance loss function.
###  We take the matrix exponent to form affine deffeomorphism transformations.
###
###  Implementations comment -
###  In order to implement the matrix exponential (and its gradient),
###  there was a need to use the batch_size in order to unstack the parametes before taking their exp.
###  For this I needed to use always the same batch size, so I possibly removed some images from
###  the training or test set, so that the number of images mod batch_size will be zero.
###
########################################################################

import os,sys

sys.path.insert(1,os.path.join(sys.path[0],'..'))
import tensorflow as tf
from atn_helpers.spatial_transformer import transformer
from atn_helpers.tranformations_helper import transfromation_parameters_regressor,transformation_regularization
from matrix_exp import expm


class alignment_transformer_network:

    def __init__(self,x,x_crop,requested_transforms,regularizations,batch_size,image_size,crop_size,num_channels,num_classes,
                 weight_stddev,activation_func,only_stn,keep_prob):
        self.X = tf.reshape(x,shape=[-1,image_size * image_size * num_channels])  #reshaping to 2 dimensions
        self.X_crop = tf.reshape(x_crop,shape=[-1,crop_size * crop_size * num_channels])  #reshaping to 2 dimensions
        self.requested_transforms = requested_transforms
        self.regularizations = regularizations
        self.image_size = image_size
        self.crop_size = crop_size
        self.num_channels = num_channels
        self.num_classes = tf.cast(num_classes,tf.int32)
        self.batch_size = batch_size
        self.only_stn = only_stn
        self.keep_prob = keep_prob
        self.weight_stddev = weight_stddev
        self.activation_func = activation_func
        self.sigma = 0.5  # for Geman-Mecclure robust function
        self.affine_maps = None
        self.x_theta = x
        self.transformations_regularizers = tf.constant([0.,0.,0.,0.,0.,0.])

    def stn_diffeo(self):

        with tf.variable_scope("atn"):
            x_tensor = tf.reshape(self.X,[-1,self.image_size,self.image_size,self.num_channels])
            theta,self.affine_maps, d2 = transfromation_parameters_regressor(self.requested_transforms,self.X_crop,
                                                                             self.keep_prob,self.crop_size,
                                                                             self.batch_size,self.num_channels,
                                                                             self.activation_func,self.weight_stddev)

            #theta = tf.Print(theta,[theta],message="theta: ",summarize=100)
            # CONVERT THETA TO BE NON-EXP
            out_size = (self.image_size,self.image_size)
            theta_exp = expm(-theta,self.batch_size)  # compute matrix exponential on {-theta}
            x_theta, d = transformer(x_tensor,theta_exp,out_size)
            #to avoid the sparse indexing warning, comment the next line, and uncomment the one after it.
            self.x_theta = tf.reshape(x_theta,shape=[-1,self.image_size,self.image_size,self.num_channels])

            #slice only the first row to show corrent status
            self.theta_first_row = (tf.slice(theta,[0,0],[1,6]))

            d.update({'params':d2['params']})

            return self.x_theta,self.affine_maps,theta, d


    def compute_alignment_loss(self,lables_one_hot=None):

        with tf.variable_scope("atn"):
            self.lables_one_hot = lables_one_hot
            if self.only_stn == True:
                print("only STN is {}!! (so no alignment...)".format(self.only_stn))
                zero = tf.constant(0.)
                return zero
            self.alignment_loss, a, b = self.alignment_loss()
            return self.alignment_loss, a, b


    def alignment_loss(self):
        # ------------------------ Our loss (with W) ------------------------------------------------------
        # #multiply W (binary mask) with the 3-dim image (tested):
        # img_slice = tf.slice(self.logits,[0,0,0,0],[-1,-1,-1,self.num_channels - 1])  # (64, 128, 128, 3)
        # #img_slice = tf.Print(img_slice,[img_slice],message="img_slice is a: ",summarize=100)
        # w_slice = tf.slice(self.logits,[0,0,0,self.num_channels - 1],[-1,-1,-1,-1])  # (64, 128, 128, 1)
        # #w_slice = tf.Print(w_slice,[w_slice],message="w_slice is a: ",summarize=100)
        # logits_new = tf.multiply(img_slice,w_slice)  # (64, 128, 128, 3)
        # #logits_new = tf.Print(logits_new,[logits_new],message="logits_new is a: ",summarize=100)
        # sum_weighted_imgs = tf.reduce_sum(logits_new,0)  # (128, 128, 3)
        # #sum_weighted_imgs = tf.Print(sum_weighted_imgs,[sum_weighted_imgs],message="sum_weighted_imgs is a: ",summarize=100)
        # sum_weights = tf.reduce_sum(w_slice,0)  # (128, 128, 1)
        # #sum_weights = tf.Print(sum_weights,[sum_weights],message="sum_weights is a: ",summarize=100)
        # sum_weights = tf.concat([sum_weights,sum_weights,sum_weights],2)  # (128, 128, 3)
        # #sum_weights = tf.Print(sum_weights,[sum_weights],message="sum_weights is a: ",summarize=100)
        #
        # # If "sum_weights" = 0 for pixel i, then "averages_new" should be 0
        # averages_new = tf.where(tf.less(sum_weights,1e-3),tf.zeros_like(sum_weighted_imgs),tf.divide(sum_weighted_imgs,sum_weights+1e-7)) # (128, 128, 3)
        # #averages_new = tf.Print(averages_new,[averages_new],message="averages_new is a: ",summarize=100)
        #
        # weighted_diff = tf.multiply(w_slice,tf.subtract(img_slice,averages_new))  # (64, 128, 128, 3)
        # #weighted_diff = tf.Print(weighted_diff,[weighted_diff],message="weighted_diff is a: ",summarize=100)
        # sum_weighted_diff = tf.reduce_sum(tf.square(weighted_diff),0)  # (128, 128, 3)
        #
        # # PRINTINGS
        # #tmp = tf.slice(sum_weighted_diff,[90,90,0],[-1,-1,-1])
        # # tmp = tf.Print(tmp,[tmp],message="sum_weighted_diff is a: ",summarize=100)
        # a = tf.reduce_max(w_slice)
        # #a = tf.Print(a,[a],message="max(img_slice): ")
        # b = tf.reduce_min(w_slice)
        # #b = tf.Print(b,[b],message="min(img_slice): ")
        #
        # # If "sum_weights" = 0 for pixel i, then "square_mean_new" should be 0
        # square_mean_new = tf.where(tf.less(sum_weights,1e-3),tf.zeros_like(sum_weighted_diff),tf.divide(sum_weighted_diff,sum_weights+1e-7))  # (128, 128, 3)
        # #square_mean_new = tf.Print(square_mean_new,[square_mean_new],message="square_mean_new is a: ",summarize=100)
        # #alignment_loss = tf.reduce_sum(square_mean_new) # tf.reduce_sum(square_mean_new) # return
        # alignment_loss = tf.reduce_sum(sum_weighted_diff)  # tf.reduce_sum(square_mean_new)
        # #alignment_loss = tf.Print(alignment_loss,[alignment_loss],message="alignment_loss is a: ",summarize=100)

        # ------------------------ Asher's loss (without W) ------------------------------------------------------
        averages = tf.reduce_mean(self.x_theta,0)
        diff = tf.subtract(self.x_theta,averages)
        sqaure_mean = tf.reduce_sum(tf.square(diff),0)  # reduce_mean
        robust_loss = sqaure_mean / (sqaure_mean + self.sigma ** 2)
        alignment_loss = tf.reduce_sum(robust_loss)  # reduce_mean
        a = tf.reduce_max(averages)
        b = tf.reduce_max(averages)

        return alignment_loss, a, b


    def compute_transformations_regularization(self,affine_maps=None):

        if self.only_stn == True:
            print("only STN is {}!! (so --currently(!)-- not calculating the reg loss...)".format(self.only_stn))
            zero = tf.constant(0.)
            return zero

        #if affine_maps is equal to None, then the user wants to compute congealng loss for a different layer than that for which
        # he ran the STN on. So he needs to give the parameter maps he got from the STN layer as an input.
        # otherwise we'll assume that on this layer he also ran the STN, and we'll already have the affine_maps in self.affine_maps.
        if affine_maps is None:
            affine_maps = self.affine_maps
        self.transformations_regularizers = transformation_regularization(affine_maps,
                                                                          self.regularizations)  #give a diffrent penalty to each type of transformation magnituted
        return self.transformations_regularizers


    def get_theta_first_row(self):
        return self.theta_first_row
########################################################################
###
###  This file creates a mnist-like dataset for a specific digit,
###  after rotating each of the specific digit's images, to
###  be alined to a fixed position.
###  This is achieved by using a pixel-location variance loss function.
###  Here we are restricting the transformations to only by rotations.
###  We also take the matrix exponent to form affine deffeomorphism
###  transformations.
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
import numpy as np
import matplotlib.pyplot as plt
import time
import mnist_helper
from ATN import alignment_transformer_network
from atn_helpers.tranformations_helper import register_gradient
from data_provider import DataProvider
from Plots import open_figure,PlotImages
from atn_helpers.spatial_transformer import transformer
from matrix_exp import expm

#from skimage.transform import warp, AffineTransform


# %% Load data
def main():
    # Here you can play with some parameters.
    digit_to_align = 4  #the digit which should we align
    n_epochs = 1
    iter_per_epoch = 7000
    batch_size = 16
    num_channels = 3

    # possible trasromations = "r","sc","sh","t","ap","us","fa"
    # see explanations in transformations_helper.py
    requested_transforms = ["t"]  #["r","t","sc","sh"]
    regularizations = {"r":0,"t":0,"sc":0,"sh":0,"fa":0}  # 0.000005
    alignment_reg = 1000

    # param my_learning_rate
    # Gets good results with 1e-4. You can also set the weigts in the transformations_helper file
    # (good results also with 1e-4 initialization)
    my_learning_rate = 1e-4
    weight_stddev = 1e-4

    activation_func = "tanh"

    #measure the time
    start_time = time.time()

    # Upload data
    img_sz = 128
    crop_size = 80
    video_name = "movies/BG.mp4"
    data = DataProvider(video_name,img_sz,crop_size,num_channels)

    device = '/cpu:0'
    with tf.device(device):  #greate the graph
        loss,x_theta,x_theta_gt,theta_first_row,theta_gt_first_row,b_s,x,x_crop,theta_gt,keep_prob,optimizer, alignment_loss, trans_regularizer, a, b, d = computational_graph(my_learning_rate,
                                                                                      requested_transforms,batch_size,
                                                                                      regularizations,activation_func,
                                                                                      weight_stddev,num_channels,
                                                                                      alignment_reg,img_sz,crop_size)

    # We now create a new session to actually perform the initialization the variables:
    params = (data,iter_per_epoch,n_epochs,batch_size,loss,x_theta,x_theta_gt,theta_first_row,theta_gt_first_row,b_s,x,x_crop,theta_gt,keep_prob,optimizer,
              start_time,img_sz,num_channels, trans_regularizer, a, b, d)
    run_session(*params)

    duration = time.time() - start_time
    print("Total runtime is " + "%02d" % (duration) + " seconds.")


def computational_graph(my_learning_rate,requested_transforms,batch_size,regularizations,activation_func,weight_stddev,
                        num_channels,alignment_reg,img_sz,crop_size):
    x = tf.placeholder(tf.float32,[None,img_sz * img_sz * num_channels])  # input data placeholder for the atn layer
    x_crop = tf.placeholder(tf.float32,[None,crop_size * crop_size * num_channels])  # input data placeholder for the atn layer
    theta_gt = tf.placeholder(tf.float32,[None,6]) # keep the real transformations of each data point x
    theta_gt_first_row = (tf.slice(theta_gt,[0,0],[1,6]))

    #batch size
    b_s = tf.placeholder(tf.float32,[1,])
    keep_prob = tf.placeholder(tf.float32)
    sigma = 0.7

    # ------------- Learn theta: ----------------
    atn = alignment_transformer_network(x,x_crop,requested_transforms,regularizations,batch_size,img_sz,crop_size,num_channels,1,
                                        weight_stddev,activation_func,False,1)
    x_theta,affine_maps,theta, d = atn.stn_diffeo()
    #theta = tf.Print(theta,[theta],message="theta: ",summarize=100)
    #x_theta = tf.layers.batch_normalization(x_theta)
    theta_first_row = atn.get_theta_first_row()
    transformations_regularizer = atn.compute_transformations_regularization(affine_maps)

    # ------------- Compute loss: ----------------
    alignment_loss, a, b = atn.compute_alignment_loss()
    alignment_loss = alignment_reg * alignment_loss
    #loss = compute_final_loss(alignment_loss,transformations_regularizer, num_channels)
    #loss = tf.reduce_sum(tf.abs(tf.subtract(theta_gt,theta)))  # !!! override loss to be on the thetas instead on x_theta
    diff = tf.subtract(theta_gt,theta)
    sqaure_sum = tf.reduce_sum(tf.square(diff),0)  # reduce_mean
    loss = tf.reduce_sum(sqaure_sum / (sqaure_sum + sigma ** 2))

    opt = tf.train.AdamOptimizer(learning_rate=my_learning_rate)
    optimizer = opt.minimize(loss)
    #grads = opt.compute_gradients(loss, [b_fc_loc2])

    a = theta_gt
    #a = tf.Print(a,[a],message="theta_gt: ",summarize=100)

    # ------------- Transform x by the ground-truth theta: -------------
    x_tensor = tf.reshape(x,[-1,height,width,num_channels])
    #x_tensor = tf.Print(x_tensor,[x_tensor],message="x_tensor: ",summarize=100)
    theta_gt_exp = expm(-theta_gt, batch_size) # compute matrix exponential on {-theta_gt}
    #theta_gt_exp = tf.Print(theta_gt_exp,[theta_gt_exp],message="theta_gt_exp: ",summarize=100)
    out_size = (height,width)
    x_theta_gt, _ = transformer(x_tensor,theta_gt_exp,out_size)
    x_theta_gt = tf.reshape(x_theta_gt,shape=[-1,height,width,num_channels])
    #x_theta_gt = tf.Print(x_theta_gt,[x_theta_gt],message="x_theta_gt: ",summarize=100)

    return loss,x_theta,x_theta_gt,theta_first_row,theta_gt_first_row,b_s,x,x_crop,theta_gt,keep_prob,optimizer, alignment_loss, transformations_regularizer, a, b, d


def compute_final_loss(alignment_loss,transformations_regularizer, num_channels):
    alignment_loss /= (height * width * num_channels)  # we need this, other wise the alignment loss is to big and we'll completely zoom out and and ruin the input image
    return alignment_loss + transformations_regularizer


def run_session(data,iter_per_epoch,n_epochs,batch_size,loss,x_theta,x_theta_gt,theta_first_row,theta_gt_first_row,b_s,x,x_crop,theta_gt,keep_prob,
                optimizer,start_time,img_sz,num_channels, trans_regularizer, a, b, d):
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    #for epoch_i in range(n_epochs):
    epoch_i = 0

    # Train step:
    for iter_i in range(iter_per_epoch):
        batch_x, batch_x_crop, batch_theta_gt = data.next_batch(batch_size, 'train')
        # T_g = d['T_g']
        # T_g = tf.Print(T_g,[T_g],message="T_g: ",summarize=100)
        # grid = d['grid']
        # grid = tf.Print(grid,[grid],message="grid: ",summarize=100)
        # theta2 = d['theta']
        # theta2 = tf.Print(theta2,[theta2],message="theta2: ",summarize=100)
        # x_s_flat = d['x_s_flat']
        # x_s_flat = tf.Print(x_s_flat,[x_s_flat],message="x_s_flat: ",summarize=100)
        # y_s_flat = d['y_s_flat']
        # y_s_flat = tf.Print(y_s_flat,[y_s_flat],message="y_s_flat: ",summarize=100)
        # params = d['params']
        # params = tf.Print(params,[params],message="params: ",summarize=100)
        loss_val, theta_first_row_val, theta_gt_first_row_val, trans_regularizer_val, a_val, b_val = sess.run([loss,theta_first_row, theta_gt_first_row, trans_regularizer, a,b],
                                      feed_dict={
                                          b_s:[batch_size],
                                          x:batch_x,
                                          x_crop:batch_x_crop,
                                          keep_prob:1.0,
                                          theta_gt:batch_theta_gt
                                      })
        if iter_i % 20 == 0:
            print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss_val))
            print("theta row 1 is: " + str(theta_first_row_val[0,:]))
            print("theta_gt row 1 is: " + str(theta_gt_first_row_val[0,:]))
        sess.run(optimizer,feed_dict={b_s:[batch_size],x:batch_x,x_crop:batch_x_crop,keep_prob:1.0,theta_gt:batch_theta_gt})

    # Test step:
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nrunning test data...")
    test_loss = 0.
    for iter_i in range(batch_size):
        batch_x, batch_x_crop, batch_theta_gt = data.next_batch(batch_size,'test')
        loss_val, x_theta_val, x_theta_gt_val, a_val2 = sess.run([loss, x_theta, x_theta_gt,a],
                                                feed_dict={
                                                    b_s:[batch_size],
                                                    x:batch_x,
                                                    x_crop:batch_x_crop,
                                                    keep_prob:1.0,
                                                    theta_gt:batch_theta_gt
                                                })
        test_loss += loss_val
    test_loss /= batch_size

    # Plot the test results:
    plot_results(batch_x, x_theta_gt_val, x_theta_val, img_sz, num_channels)

    # Print loss of tests data:
    print('Alignment loss (%d): ' % (epoch_i + 1) + str(test_loss) + "\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    if np.isnan(test_loss):
        duration = time.time() - start_time
        print("Total runtime is " + "%02d" % (duration) + " seconds.")
        raise SystemExit

    sess.close()


def plot_results(batch_x, x_theta_gt_val, x_theta_val, img_sz, num_channels):
    imgs_x = []
    imgs_x_theta_gt = []
    imgs_x_theta = []
    titles = []
    panoramas = []
    #panoramic_gt = np.zeros((img_sz,img_sz,num_channels))
    #panoramic_test = np.zeros((img_sz,img_sz,num_channels))
    for i in range(10):
        I = np.reshape(batch_x[i,...],(img_sz,img_sz,num_channels))
        imgs_x.append(I[:,:,0:3])
        I = np.reshape(x_theta_gt_val[i,...],(img_sz,img_sz,num_channels))
        imgs_x_theta_gt.append(np.abs(I[:,:,0:3]))
        #panoramic_gt = np.concatenate((panoramic_gt,np.abs(I[:,:,0:3])),axis=2)
        I = np.reshape(x_theta_val[i,...],(img_sz,img_sz,num_channels))
        imgs_x_theta.append(np.abs(I[:,:,0:3]))
        #panoramic_test = np.concatenate((panoramic_test,np.abs(I[:,:,0:3])),axis=2)
        titles.append('')

    fig1 = open_figure(1,'Input Data (x)',(7,3))
    PlotImages(1,2,5,1,imgs_x,titles,'gray',axis=True,colorbar=False)
    fig2 = open_figure(2,'Ground Truth Theta Warp',(7,3))
    PlotImages(2,2,5,1,imgs_x_theta_gt,titles,'gray',axis=True,colorbar=False)
    fig3 = open_figure(3,'Learned Theta Warp',(7,3))
    PlotImages(3,2,5,1,imgs_x_theta,titles,'gray',axis=True,colorbar=False)

    # build panoramic image:
    panoramic_gt = np.nanmean(nan_if(imgs_x_theta_gt),axis=0)
    panoramic_test = np.nanmean(nan_if(imgs_x_theta),axis=0)
    panoramas.append(panoramic_gt)
    panoramas.append(panoramic_test)
    fig4 = open_figure(4,'Panoramic Image (GT and Test)',(6,4))
    PlotImages(4,1,2,1,panoramas,titles,'gray',axis=True,colorbar=False)

    plt.show()
    fig1.savefig('1_X.png', dpi=1000)
    fig2.savefig('2_GT Theta.png', dpi=1000)
    fig3.savefig('3_Theta.png', dpi=1000)
    fig4.savefig('4_Panorama.png', dpi=1000)


def nan_if(lst):
    new_lst = []
    for i in range(len(lst)):
        I = lst[i]
        I_new = np.where(I <= 1e-04, np.nan, I)
        #I_new = np.where(I_new == 1, np.nan, I_new)
        new_lst.append(I_new)
    return new_lst


if __name__ == '__main__':
    #register the gradient for matrix exponential
    register_gradient()
    height,width = [128,128]
    main()
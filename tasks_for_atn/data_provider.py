import numpy as np
import os
import shutil
import glob
import cv2
import random
import matplotlib.pyplot as plt
import math
from scipy.linalg import logm
import matplotlib.pyplot as plt


class DataProvider:

    def __init__(self, video_name, img_sz, crop_size, num_channels):

        # self.config = config
        self.feed_path = "../Data"

        # load data from video
        # makedir(self.feed_path)
        # feed_size = extractImages(video_name, self.feed_path)
        #feed_size = 8900

        s_idx = img_sz // 2 - crop_size // 2

        print('Start loading data ...')

        # prepare cropped images from all data set
        files = glob.glob(self.feed_path + "/*.jpg")
        self.train = []
        self.train_crop = []
        self.train_theta = []
        self.test = []
        self.test_crop = []
        self.test_theta = []
        t_x = 0
        t_y = 0

        for img_str in files:
            if t_x+crop_size <= img_sz and t_y+crop_size <= crop_size:

                # define euclidean params (t1,t2,r)
                t_x = random.randint(0, 35)
                t_y = random.randint(0, 35)
                rot = np.deg2rad(random.randint(0,30))

                I = cv2.imread(img_str)
                I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
                I = cv2.resize(I, (img_sz, img_sz))

                # if there is a rotation:
                M = cv2.getRotationMatrix2D((img_sz / 2,img_sz / 2),np.rad2deg(rot),1)
                I = cv2.warpAffine(I,M,(img_sz,img_sz), cv2.INTER_LANCZOS4)

                crop = I[t_y:t_y+crop_size, t_x:t_x+crop_size, :]
                crop = crop / 255.

                # ----- generate outliers in a low probability:
                # r = random.uniform(0, 1)
                # if r >= 0.70:
                #     r2 = int(random.uniform(0, 1)*(crop_size/2))
                #     crop = generate_outliers(crop, r2, r2+30)

                # ----- prepare the embedded image:
                img = np.zeros((img_sz,img_sz,3))
                img[0:crop_size,0:crop_size,:] = crop
                img = np.reshape(img,(img_sz * img_sz * num_channels))
                crop = np.reshape(crop,(crop_size * crop_size * num_channels))

                # ----- create the ground-truth theta: translation & rotation:
                theta_gt = np.zeros((6))
                theta_gt[2] = float(t_x) / 64
                theta_gt[5] = float(t_y) / 64
                theta_gt[1] = -rot
                theta_gt[3] = rot

                self.train_crop.append(crop)
                self.train.append(img)
                self.train_theta.append(theta_gt)
                self.test.append(img)
                self.test_crop.append(crop)
                self.test_theta.append(theta_gt)

            else:
                t_x = 0
                t_y = 0

        self.train = np.array(self.train)
        self.train_crop = np.array(self.train_crop)
        self.train_theta = np.array(self.train_theta)
        self.test = np.array(self.test)
        self.test_crop = np.array(self.test_crop)
        self.test_theta = np.array(self.test_theta)
        self.train_size = self.train.shape[0]
        self.test_size = self.test.shape[0]

        print('Finished uploading data, Train data shape:', self.train.shape, '; Test data shape:', self.test.shape)

    def next_batch(self, batch_size, data_type):
        batch_x = None
        batch_theta_gt = None
        if data_type == 'train':
            idx = np.random.choice(self.train_size, batch_size) # np.arange(25)
            batch_x = self.train[idx, ...]
            batch_x_crop = self.train_crop[idx, ...]
            batch_theta_gt = self.train_theta[idx, ...]
        elif data_type == 'test':
            idx = np.random.choice(self.test_size, batch_size)  # np.arange(25)
            batch_x = self.test[idx, ...]
            batch_x_crop = self.test_crop[idx,...]
            batch_theta_gt = self.test_theta[idx,...]
        return batch_x, batch_x_crop, batch_theta_gt


def generate_outliers(X, s, e):
    X_o = np.reshape(X, X.shape)
    start_idx = np.array([s, s])
    end_idx = np.array([e, e])
    for i in range(start_idx[0], end_idx[0]):
        for j in range(start_idx[1], end_idx[1]):
            X_o[i][j] = np.random.random_integers(0, 1)
    return X_o


def makedir(folder_name):
    try:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    except OSError:
        pass
    # cd into the specified directory
    # os.chdir(folder_name)
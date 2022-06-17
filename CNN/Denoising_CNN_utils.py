import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization, Conv2D, PReLU, Dropout
from keras.initializers import Constant
import cv2 as cv
import re
'''This file contains various functions used both by Denoising_CNN_main.py'''

def load_data_noisy(load_path, file_names, pixels):
    print ("loading data...")
    #print (file_names)
    #number of images should be the same for X and Y
    num_im = len(file_names)
    #Then I load the other images in the array train_X
    images = np.zeros((num_im, pixels, pixels,1))
    r = list(range(num_im))
    #random.shuffle(r)
    label = 0
    for i in r:
        #first I read the header to know what type of defect it is to make the Y_vector
        file_name = load_path+file_names[i]
        f = open(file_name)
        label_str_regex = re.compile(r'label_(\d+)')
        label_temp = label_str_regex.search(file_name).group()
        label = int(re.findall(r'\d+', label_temp)[0])
        images[label,:,:,0] = np.loadtxt(file_name, max_rows = pixels)
    return images

#Function to load data
def my_load_data_to_predict(open_path_X,load_path_X, file_names_X, pixels_X):
    print (file_names_X)
    #number of images should be the same for X and Y
    num_im = len(file_names_X)
    #Then I load the other images in the array train_X
    X = np.zeros((num_im, pixels_X, pixels_X,1))
    for i in range(num_im):
        file_name_X = load_path_X+file_names_X[i]
        X[i,:,:,0] = np.loadtxt(file_name_X, max_rows = pixels_X)
    return X

#Function where the models are defined
#Model n°3 was used in the paper
def get_uncompiled_model_filter(model_number, dropout_rate, pixels, num_filters):
    if model_number == 1:
        model = models.Sequential()
        #put some layers
        #model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), input_shape=(pixels, pixels, 1), padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(BatchNormalization())
        model.add(Conv2D(1, (3, 3), padding='same', activation='tanh'))
    if model_number == 2:
        model = models.Sequential()
        #put some layers
        model.add(Conv2D(num_filters, (3, 3), input_shape=(pixels, pixels, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(1, (3, 3), padding='same', activation='tanh'))
    if model_number == 3:
        model = models.Sequential()
        #put some layers
        #block1 down
        model.add(Conv2D(num_filters, (3, 3), input_shape=(pixels, pixels, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.MaxPooling2D((2, 2)))
        #block 2 down
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.MaxPooling2D((2, 2)))
        #lower block
        model.add(Conv2D(4*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(4*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.UpSampling2D())
        #block 1 up
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(2*num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.UpSampling2D())
        #block 2 up
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        #final
        model.add(Conv2D(1, (3, 3), padding='same', activation='tanh'))

    if model_number == 4:
        model = models.Sequential()
        #put some layers
        #block1 down
        model.add(Conv2D(num_filters, (3, 3), input_shape=(pixels, pixels, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.MaxPooling2D((2, 2)))
        #block 2 down
        model.add(Conv2D(2*num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(2*num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.MaxPooling2D((2, 2)))
        #lower block
        model.add(Conv2D(4*num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(4*num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.UpSampling2D())
        #block 1 up
        model.add(Conv2D(2*num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(2*num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.UpSampling2D())
        #block 2 up
        model.add(Conv2D(num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        #final
        model.add(Conv2D(1, (3, 3), padding='same', activation='tanh'))
    return model

#returns the compiled model
def get_compiled_model(model_number, loss, alpha, dropout_rate, pixels, learning_rate, num_filters):
    model = get_uncompiled_model_filter(model_number, dropout_rate, pixels, num_filters)
    print("Fred is compiling the model...")
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if loss == "MAE":
        model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=opt, metrics=['mean_absolute_error'])
    if loss == "MSE":
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer= opt, metrics=['mean_squared_error'])
    if loss == "SSIM":
        model.compile(loss=my_loss_function_SSIM(alpha), optimizer=opt, metrics=['mean_absolute_error'])
    if loss == "SSIMm":
        model.compile(loss=my_loss_function_SSIM_multi(alpha), optimizer= opt, metrics=['mean_absolute_error'])
    return model

#Defines a SSIM loss function
def my_loss_function_SSIM(alpha):
    def my_loss(im_actual,im_predicted):
        #first I subtract the minimum to each image
        min_ = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.reduce_min(im_predicted, axis = [1,2,3]), axis=1), axis=2),axis=3)
        im_predicted = -2*min_+im_predicted
        min_ = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.reduce_min(im_actual, axis = [1,2,3]), axis=1), axis=2),axis=3)
        im_actual = -2*min_+im_actual
        max_val_ = tf.reduce_max(im_predicted)
        MAE = tf.reduce_mean(tf.abs(im_actual - im_predicted), axis=(1, 2, 3))
        SSIM = tf.image.ssim(im_actual, im_predicted, max_val=max_val_, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        return [(1-alpha)*MAE-alpha*SSIM]
    return my_loss

#Defines a SSIM m loss function.
def my_loss_function_SSIM_multi(alpha):
    def my_loss(im_actual,im_predicted):
        #first I subtract the minimum to each image
        min_ = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.reduce_min(im_predicted, axis = [1,2,3]), axis=1), axis=2),axis=3)
        im_predicted = -2*min_+im_predicted
        min_ = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.reduce_min(im_actual, axis = [1,2,3]), axis=1), axis=2),axis=3)
        im_actual = -2*min_+im_actual
        max_val_ = tf.reduce_max(im_predicted)
        MAE_ = tf.reduce_mean(tf.abs(im_actual - im_predicted), axis=(1, 2, 3))
        #SSIM_multi = tf.image.ssim_multiscale(im_actual, im_predicted, max_val=max_val_, filter_size=8, filter_sigma=1.5, k1=0.01, k2=0.03)
        SSIM_multi = tf.image.ssim_multiscale(im_actual, im_predicted, max_val=max_val_, filter_size=8, filter_sigma=1.5, k1=0.1, k2=0.3)
        return [(1-alpha)*MAE_-alpha*SSIM_multi]
    return my_loss
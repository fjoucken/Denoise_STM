import tensorflow as tf
import winsound
import numpy as np
import matplotlib.pyplot as plt
import os.path
import glob
import datetime

from Denoising_CNN_utils import  load_data_noisy, get_compiled_model

'''This code trains a CNN for denoising scanning tunneling microscopy images. The inputs are noisy images
and the labels are the corresponding non-noisy images.'''

#Checking your GPUs
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Getting the time for naming of the model
x = datetime.datetime.now()

#number of pixels in the images
pixels = 64
#number of filters in upper blocks
num_filters = 32
#The CNN model you chose (see Denoising_CNN_utils.py)
#number 3 was used in the paper.
model_num = 3
#number of epochs
no_epochs = 150
#batch size
batch_size = 20
#Drop out rate
drop_out_rate = 0.0
#learning rate
learning_rate = 0.001
#loss (MAE works well)
loss = "MAE"   #choose MAE or SSIM or SSIMm
#Parameter to set if you use SSIM as a loss (see this paper for more info: https://aip.scitation.org/doi/10.1063/5.0054920)
alpha = 1 #for SSIM (1 is pure SSIM, 0 is MAE)
#I name the model for saving
model_name = x.strftime("%m_%d_%y_%H_%M_%S_")+"Denoising_pristine_"+loss+"_alpha_"+str(alpha)+"_LR_"+str(learning_rate)+"_filter_"+str(num_filters)+"_DO_"+str(drop_out_rate)+"_epochs_"+str(no_epochs)+"_model_"+str(model_num)

#Here I specify the name of the folder where to get the training data
folder_name = 'final'
#Then I get the training data folders
base_name_train_X = 'D:/Machine_learning/Generated_training_data/Denoising/Final/'+folder_name+'/train/X/'
base_name_train_Y = 'D:/Machine_learning/Generated_training_data/Denoising/Final/'+folder_name+'/train/Y/'
#And the test data folder (for validation)
base_name_test_X = 'D:/Machine_learning/Generated_training_data/Denoising/Final/'+folder_name+'/test/X/'
base_name_test_Y = 'D:/Machine_learning/Generated_training_data/Denoising/Final/'+folder_name+'/test/Y/'

#Now I load the training data
#First the input
open_path = base_name_train_X+'*.txt'
file_names = [os.path.basename(x) for x in glob.glob(open_path)]
print ("loading train_X")
train_X = load_data_noisy(base_name_train_X, file_names, pixels)
#Then the labels
open_path = base_name_train_Y+'*.txt'
file_names = [os.path.basename(x) for x in glob.glob(open_path)]
print ("loading train_Y")
train_Y = load_data_noisy(base_name_train_Y, file_names, pixels)
print ("train_X shape is:", train_X.shape)
print ("train_Y shape is:", train_Y.shape)

#same for testing data
open_path = base_name_test_X+'*.txt'
file_names = [os.path.basename(x) for x in glob.glob(open_path)]
print ("loading test_X")
test_X = load_data_noisy(base_name_test_X, file_names, pixels)
open_path = base_name_test_Y+'*.txt'
file_names = [os.path.basename(x) for x in glob.glob(open_path)]
print ("loading test_Y")
test_Y = load_data_noisy(base_name_test_Y, file_names, pixels)
print ("test_Y shape is:", test_Y.shape)

#Now I create the model
model = get_compiled_model(model_number = model_num, 
                            loss = loss,
                            alpha = alpha,
                            dropout_rate = drop_out_rate, 
                            pixels = pixels,
                            learning_rate = learning_rate,
                            num_filters = num_filters)

model.summary()
#This is your path to save the mdoel
base_name = "D:/Machine_learning/My_models/Denoising_CNN_STM/"
checkpoint_path = base_name+model_name+".ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#Now I fit the model
print("We are fitting the model...")
history = model.fit(train_X, 
                    train_Y, 
                    epochs=no_epochs, 
                    batch_size=batch_size, 
                    shuffle=True,
                    validation_data=(test_X, test_Y),
                    callbacks=[cp_callback])

#For updating checkpoints files at the end of each epoch:
os.listdir(checkpoint_dir) #not sure it is used

#A beep to wake you up!
winsound.Beep(800, 1000)

#Then you plot the history
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label = 'Val. loss')
print("history:",history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
#And you save the history
file_name = base_name+x.strftime("%m_%d_%y_%H_%M_%S_")+'loss.txt'
np.savetxt(file_name, history.history['loss'])
file_name = base_name+x.strftime("%m_%d_%y_%H_%M_%S_")+'val_loss.txt'
np.savetxt(file_name, history.history['val_loss'])

#I print the test loss and accuracy, for info.
test_loss, test_acc = model.evaluate(test_X,  test_Y, verbose=2)
print ("test_loss and test_accuracy are:")
print (test_loss, test_acc)
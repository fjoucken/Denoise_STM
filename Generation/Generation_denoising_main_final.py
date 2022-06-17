from configparser import NoSectionError
import winsound #to wake you up when the computation is done!
import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
import matplotlib
#from utils_model import doped_graphene
from Generation_denoising_utils_MLG_final import few_dopants_distri_MLG, ldos_from_params_MLG
from Generation_denoising_utils_BLG_final import few_dopants_distri_BLG, ldos_from_params_BLG
from Generation_denoising_utils_STMimage_final import make_STM_image_noisy_strain, add_paraboloid, save_image

################################################################################################################################
################################################################################################################################
################################################################################################################################
# This script generates a number of simulated STM images of mono- or bi-layer graphene with two types of noise
# dz is the hieght noise and dx is the line noise (see github repo). The label is the same STM image with no noise
################################################################################################################################
################################################################################################################################
################################################################################################################################

#I start defining parameters
#folder where to put the data (typically train or test or apply)
train_or_test = "train"
#The path for saving:
base_name = 'D:/Machine_learning/Generated_training_data/Denoising/Final/pristine_case/'
save_path_X = base_name+train_or_test+"/X/"
save_path_Y = base_name+train_or_test+"/Y/"
save_path_X_jpeg = base_name+train_or_test+"/X_jpeg/"
save_path_Y_jpeg = base_name+train_or_test+"/Y_jpeg/"
#if true, you save a png version of the image
save_png = True
#label of the first image. Should be 0 except if you add images to preexisting set.
label = 1000
#number of models (LDOS) that will be computed
num_of_models = 4000 
#num of STM images for each model
num_of_STM_images = 1 
#num of noisy images per clean STM image
num_of_noisy_images = 1 
#minimum and maximum lateral size of the images (in nm)
l_STM_min, l_STM_max = 1, 4     
#number of pixels in the images        
pxls = 64 

#Setting the parameters for the noise
# dx is the "line noise"; 0.1 works 
dx_min, dx_max = 0.0, 0.1
#dz is the height noise
dz_min, dz_max = 0.00, 0.07
#strain coeficcient, if you want to add some strain
strain_max = 0.0

#This give the dopant distribution probabolity. The first argument gives probability of having no dopant on the image.
#the second argument gives the probability of having 1 dopant and the second arg 2 dopants.
#p_dopants = [0.35, 0.35, 0.3]
p_dopants = [1, 0, 0]
#Probability of having zero vacancy or 1 vacancy (I didn't use it)
p_vac = [1, 0] 
#Z range for the generation of the STM images
#minimum and maximum height
z_min, z_max = 0.6, 1.5      
#Energy range for the generation of the STM images
#minimum and maximum energy of interest
E_oi_min, E_oi_max = -0.5, 0.5

#parameters for background (see function add_paraboloid):
x_c, y_c, hyper, inv, a_b_ratio_max, amp_max = 0, 0, 1, 1, 3, 2

#max angle of the lattice. Angle is chosen between 0 and this max angle
theta_max = 360
#the broadening in the LDOS calculation         
g = 0.1                 
#Onsite potential of the dopant
V = -10   
#Energy range within which the LDOS is computed              
E_range = 3 
#Energy resolution of the LDOS
E_reso = 100
#decay of the Gaussian around the dopant; If larger, nearest neighbors of 
# the dopant have alos their onsite potential changed
sig = 0.0001              

#color for saving the image
cmap_ = 'inferno'

# [nm] graphene unit cell length
a = 0.24595   

#I count the time
tic = time.time()

#I do 3 loops: one on the models, and within that loop, one for the STM image, and one for the noisy STM image
for i in range (num_of_models):
    print ('model number:')
    print (i) 
    #set the angle of the lattice
    theta = np.radians(np.random.randint(0, high = theta_max + 1)) 
    #set the number of dopants
    num_dopants = 2*np.random.choice([0, 1, 2], p = p_dopants)
    #set the number of vacancies
    num_vac = np.random.choice([0, 1], p = p_vac)
    #choose if it is MLG or BLG
    num_of_layers = np.random.randint(1, high = 2 + 1)
    #set the size of the STM image
    l_STM = np.random.uniform(low = l_STM_min, high = l_STM_max)             
    #Now I separate according to the number of layers.
    #for MLG
    if num_of_layers == 1:
        # the shift is for ensuring there is not always an atom right at the center
        shift_x = np.random.uniform(low = -0.25, high = 0.25) * a  
        shift_y = np.random.uniform(low = -0.25, high = 0.25) * a 
        #I get the positions of the dopants/vacancies
        pos_dopants, pos_vac = few_dopants_distri_MLG(l_STM, num_dopants, num_vac, theta, shift_x, shift_y) 
        #I solve the model and get the ldos in one big function:
        ldos, _ = ldos_from_params_MLG(pos_dopants, pos_vac, theta, shift_x, shift_y, l_STM, U = V, sig = sig, E_range = E_range, E_reso = E_reso, gamma = g)
        #Now I loop for making the STM images:
        for j in range(num_of_STM_images):
            print ('STM image number:')
            print (j) 
            #I set the height 
            z = np.random.uniform(low = z_min, high = z_max)
            #I set the energy 
            E_oi = np.random.uniform(low = E_oi_min, high = E_oi_max)
            #This is the label image
            STM_image_Y = make_STM_image_noisy_strain(ldos, E_range, E_oi, pxls, l_STM, z, dx = 0, dz = 0, sx = 1, sy = 1)
            #Now I loop over the number of noisy images:
            for k in range (num_of_noisy_images):
                #I set the noise amplitude
                dx = np.random.uniform(low = dx_min, high = dx_max)
                dz = np.random.uniform(low = dz_min, high = dz_max)
                sx = np.random.uniform(low = 1-strain_max, high = 1+strain_max)
                sy = np.random.uniform(low = 1-strain_max, high = 1+strain_max)
                #this is the string for the file_name
                parameters_string_X = "_MLG_num_dopants_"+str(num_dopants)+"_dz_"+str(np.round(dz,4))+"_"\
                    "dx_"+str(np.round(dx,4))+"_l_"+str(np.round(l_STM,2)).zfill(3)+"_theta_"+str(np.round(np.degrees(theta),1))+"_"\
                    "_V_"+str(V).zfill(3)+"_E_"+str(np.round(E_oi,2)).zfill(3)+"_z_"+str(np.round(z,2))
                parameters_string_Y = "_MLG_num_dopants_"+str(num_dopants)+"_dz_"+str(np.round(dz,4))+"_"\
                    "dx_"+str(np.round(dx,4))+"_"\
                    "_l_"+str(np.round(l_STM,2)).zfill(3)+"_theta_"+str(np.round(np.degrees(theta),1))+\
                    "_V_"+str(V).zfill(3)+"_E_"+str(np.round(E_oi,2)).zfill(3)+"_z_"+str(np.round(z,2))
                #I make the noisy STM image
                STM_image_noisy = make_STM_image_noisy_strain(ldos, E_range, E_oi, pxls, l_STM, z, dx, dz, sx, sy)
                #I set the params for the background:
                x_c = np.random.uniform(low = -l_STM/2, high = l_STM/2)
                y_c = np.random.uniform(low = -l_STM/2, high = l_STM/2)
                hyper = np.random.randint(0, high = 1 + 1)
                inv = np.random.randint(0, high = 1 + 1)
                a_b_ratio = np.random.uniform(low = 1/a_b_ratio_max, high = a_b_ratio_max)
                amp = np.random.uniform(low = 0, high = amp_max)
                #I add the background
                STM_image_noisy = add_paraboloid(STM_image_noisy, l_STM, pxls, x_c, y_c, hyper, inv, a_b_ratio, amp)
                #I also add the background to the label image
                STM_image_Y = add_paraboloid(STM_image_Y, l_STM, pxls, x_c, y_c, hyper, inv, a_b_ratio, amp)
                #then I save the image
                save_image(label, save_path_X, save_path_X_jpeg, parameters_string_X, STM_image_noisy, save_png, cmap_)
                #And save the label Y
                save_image(label, save_path_Y, save_path_Y_jpeg, parameters_string_Y, STM_image_Y, save_png, cmap_)
                label += 1
    #If it is BLG:
    if num_of_layers == 2:
        #the shift is for ensuring there is not always an atom right at the center
        shift_x = np.random.uniform(low = -0.25, high = 0.25) * a  
        shift_y = np.random.uniform(low = -0.25, high = 0.25) * a 
        #I get the positions of the dopants/vacancies
        pos_dopants, pos_vac = few_dopants_distri_BLG(l_STM, num_dopants, num_vac, theta, shift_x, shift_y) 
        #I solve the model and get the ldos in one big function:
        ldos = ldos_from_params_BLG(pos_dopants, theta, shift_x, shift_y, l_STM, U = V, sig = sig, E_range = E_range, E_reso = E_reso, gamma = g)
        #Now I loop for making the STM images:
        for j in range(num_of_STM_images):
            print ('STM image number:')
            print (j) 
            #I set the height randmoly between 0.5 and 1.5
            z = np.random.uniform(low = z_min, high = z_max)
            #I set the energy randomly between 0.1 and 1
            E_oi = np.random.uniform(low = E_oi_min, high = E_oi_max)
            #This is the label image
            STM_image_Y = make_STM_image_noisy_strain(ldos, E_range, E_oi, pxls, l_STM, z, dx = 0, dz = 0, sx = 1, sy = 1)
            #I loop on the number of noisy images
            for k in range (num_of_noisy_images):
                #I set the noise amplitude
                dx = np.random.uniform(low = dx_min, high = dx_max)
                dz = np.random.uniform(low = dz_min, high = dz_max)
                sx = np.random.uniform(low = 1-strain_max, high = 1+strain_max)
                sy = np.random.uniform(low = 1-strain_max, high = 1+strain_max)
                #set the name of the files
                parameters_string_X = "_BLG_num_dopants_"+str(num_dopants)+"_dz_"+str(np.round(dz,4))+"_"\
                    "dx_"+str(np.round(dx,4))+"_l_"+str(np.round(l_STM,2)).zfill(3)+"_theta_"+str(np.round(np.degrees(theta),1))+"_"\
                    "_V_"+str(V).zfill(3)+"_E_"+str(np.round(E_oi,2)).zfill(3)+"_z_"+str(np.round(z,2))
                parameters_string_Y = "_BLG_num_dopants_"+str(num_dopants)+"_dz_"+str(np.round(dz,4))+"_"\
                    "dx_"+str(np.round(dx,4))+"_"\
                    "_l_"+str(np.round(l_STM,2)).zfill(3)+"_theta_"+str(np.round(np.degrees(theta),1))+\
                    "_V_"+str(V).zfill(3)+"_E_"+str(np.round(E_oi,2)).zfill(3)+"_z_"+str(np.round(z,2))
                #I compute the STM image
                STM_image_noisy = make_STM_image_noisy_strain(ldos, E_range, E_oi, pxls, l_STM, z, dx, dz, sx, sy)
                #I set the params for the background:
                x_c = np.random.uniform(low = -l_STM/2, high = l_STM/2)
                y_c = np.random.uniform(low = -l_STM/2, high = l_STM/2)
                hyper = np.random.randint(0, high = 1 + 1)
                inv = np.random.randint(0, high = 1 + 1)
                a_b_ratio = np.random.uniform(low = 1/a_b_ratio_max, high = a_b_ratio_max)
                amp = np.random.uniform(low = 0, high = amp_max)
                #I add the background
                STM_image_noisy = add_paraboloid(STM_image_noisy, l_STM, pxls, x_c, y_c, hyper, inv, a_b_ratio, amp)
                #I also add the background to the Y image
                STM_image_Y = add_paraboloid(STM_image_Y, l_STM, pxls, x_c, y_c, hyper, inv, a_b_ratio, amp)
                #then I save the image
                save_image(label, save_path_X, save_path_X_jpeg, parameters_string_X, STM_image_noisy, save_png, cmap_)
                #And save the label Y
                save_image(label, save_path_Y, save_path_Y_jpeg, parameters_string_Y, STM_image_Y, save_png, cmap_)
                label += 1
toc = time.time()
winsound.Beep(800, 1000)
print("Time for computing all this stuff:")
print(toc-tic)
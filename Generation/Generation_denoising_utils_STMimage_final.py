import numpy as np
import os
import matplotlib
from math import pi

def make_STM_image_noisy_strain(LDOSes, E_range, E_oi, pxls, L, z, dx, dz, sx, sy):
    #This is constant height image with noise
    #3 noises:
    #dx variation of x at each line (to make the image having an offset at each line)
    #dz variation in z for the noise in the scanner
    #E_range gives the range in which the LDOSes are computes (from -E_range to +E_range)
    #LDOSes are the LDOSes at various energies [XYZ] and the third dimension is the energy
    #sx and sy are the strain in x and y directions. It only extends or shrink the image in x and y
    x_grid = np.linspace((-L/2)*sx, sx*L/2, pxls)
    y_grid = np.linspace(-L/2, L/2, pxls)
    c = 3/(5.29e-2) #constant used in the computation of Psi (3/(5.29e-11) in meters-1)
    E_step = (2*E_range)/(LDOSes.shape[0]-1)
    index_of_E_zero = int(np.around((LDOSes.shape[0]-1)/2)) #gives the index of energy = 0
    i_max = int(np.around(np.absolute(E_oi + E_range)/E_step,0))
    num_of_atoms = LDOSes.shape[1]
    sum_LDOS = np.zeros((num_of_atoms,1))
    LDOS_temp = np.zeros((num_of_atoms,1))
    #First I sum the LDOS up to the energy of interest
    if E_oi > 0:
        for i in range(index_of_E_zero,i_max+1):
            #LDOS_temp = LDOSes[i,:,2]
            LDOS_temp = LDOSes[i,:,3]
            sum_LDOS = np.transpose(np.add(sum_LDOS.T,LDOS_temp[:,].T))
    else:
        for i in range(i_max, index_of_E_zero+1):
            #LDOS_temp = LDOSes[i,:,2]
            LDOS_temp = LDOSes[i,:,3]
            sum_LDOS = np.transpose(np.add(sum_LDOS.T,LDOS_temp[:,].T))
            
    STM_image = np.zeros((pxls,pxls))
    temp = np.zeros((num_of_atoms,1))
    temp_x = np.zeros((num_of_atoms,1))
    temp_y = np.zeros((num_of_atoms,1))
    temp_z = np.zeros((num_of_atoms,1))
    temp_theta = np.zeros((num_of_atoms,1))
    #I define the state of the tip here
    for i in range(pxls):
        #I change dx_ at every line
        dx_ = np.random.uniform(low = 0, high = dx)
        for j in range(pxls):
            #I change dz_ at each pixel
            temp_x = (x_grid[i])-LDOSes[1,:,0]
            temp_y = (y_grid[j]+dx_)-LDOSes[1,:,1]
            temp_theta = np.arctan2(temp_x,temp_y)
            temp_z = z-LDOSes[1,:,2]
            temp = np.square(temp_x)+np.square(temp_y)
            temp += np.square(temp_z)
            temp = np.exp(-c*np.sqrt(temp))
            temp *= (pi**0.5)*(c**2.5)*z
            temp = np.square(temp)
            temp = np.multiply(sum_LDOS[:,0],temp)
            #STM_image[pxls-j-1][i] = np.sum(temp,axis = 0, keepdims = True)
            STM_image[i][j] = np.sum(temp,axis = 0, keepdims = True)
    max_ = np.max(STM_image)
    min_ = np.min(STM_image)
    scaled_image = ((STM_image - min_)/(max_ - min_))
    noise = np.random.normal(0, dz, scaled_image.shape)
    avg = np.average(scaled_image)
    STM_image = scaled_image - avg + noise
    return STM_image

def add_paraboloid(STM_image, l_STM, pixels, x_c, y_c, hyper, inv, a_b_ratio, amp):
    #this function adds a paraboloid to an STM image to simulate the bumps seen in experimental images.
    #l_STM is the size of the STM image
    #x_c and y_c are the center of the paraboloid
    #a_ratio defines the curvature ratio between x and y.
    #if hyper = 1, it is a hyperbolic paraboloid https://en.wikipedia.org/wiki/Paraboloid
    #if inv = -1, it is upside-down
    #amp defines the amplitude of the backgroun (note STM_image is supposed to be normalized)
    x = np.arange(-l_STM/2, l_STM/2, l_STM/(pixels))
    y = x
    xx, yy = np.meshgrid(x, y)
    a = 1
    b = a * a_b_ratio
    z = (-1)**inv * (((xx - x_c)/a)**2 + (-1)**hyper * ((yy - y_c)/b)**2)
    z = scale_and_noise(z, 0, amp)
    STM_image += z
    return scale_and_noise(STM_image,0, 1)

def scale_and_noise(STM_image, noise_amplitude, amplitude):
    max_ = np.max(STM_image)
    min_ = np.min(STM_image)
    scaled_image = amplitude * ((STM_image - min_)/(max_ - min_))
    noise = np. random. normal(0, noise_amplitude, scaled_image.shape)
    avg = np.average(scaled_image)
    scaled_noisy_image = scaled_image - avg + noise
    return scaled_noisy_image

def save_image(label, save_path, save_path_jpeg, parameters_string, image, save_png, cmap_):
    base_name = "STM_X_label_"+str(label)+parameters_string 
    name_of_file = base_name
    completeName = os.path.join(save_path, name_of_file+".txt")
    file = open(completeName,'wb')
    np.savetxt(file, image, header='label='+str(label),delimiter='\t')
    file.close()
    #then save the png
    if save_png == True:
        completeName = os.path.join(save_path_jpeg, name_of_file+".png")
        matplotlib.image.imsave(completeName, image, cmap=cmap_)
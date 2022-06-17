#This function gives the ldos from a model
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
import time
import io
import os.path
import matplotlib

from math import sqrt, pi
def rectangle(width, height):
        x0 = width / 2
        y0 = height / 2
        return pb.Polygon([[x0, y0], [x0, -y0], [-x0, -y0], [-x0, y0]])

def ldos_from_params_MLG(pos_dopants, pos_vac, theta, shift_x, shift_y, l_STM, U, sig, E_range, E_reso, gamma):
    #First I define the model
    coupling = 3.16
    l_model = l_STM + 15
    #l_model = l_STM + np.random.randint(2, high = 10 + 1)
    a = 0.24595   # [nm] unit cell length
    a_cc = a/(sqrt(3))  # [nm] carbon-carbon distance

    #I define the basis vectors
    a1 = np.array([a, 0])
    a2 = np.array([a/2, a/2 * sqrt(3)])
    sub_B = np.array([0, a_cc])

    #I rotate the lattice vectors by theta
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]])
    a1 = np.matmul(rot_matrix, a1)
    a2 = np.matmul(rot_matrix, a2)
    sub_B = np.matmul(rot_matrix, sub_B)
    shift = np.array([shift_x, shift_y])

    #First I define the lattice
    def monolayer_graphene():
        t0 = coupling      # [eV] nearest neighbour hopping
        lat = pb.Lattice(a1,
                        a2)
        lat.add_sublattices(('A', np.add(sub_B, shift)),
                            ('B', np.add([0, 0],shift)))
        lat.add_hoppings(
            # inside the main cell
            ([0,  0], 'A', 'B', t0),
            # between neighboring cells
            ([1, -1], 'B', 'A', t0),
            ([0, -1], 'B', 'A', t0),
        )
        return lat
    '''
    lattice = monolayer_graphene()
    lattice.plot()
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.show()
    '''
    #here is a potential with exponential decay
    #it is located on position (2 coord), contained in the list position, with a width of sig (cf. Lambin et al PRB 2012), and onsite potential U
    def dopant(positions, sigma, V):
        @pb.onsite_energy_modifier
        def potential(x, y):
            length = len(positions)
            pot = 0
            for i in range(length):
                x0, y0 = positions[i]
                pot += np.exp(-0.5*((x-x0)**2+(y-y0)**2)/(sigma**2))
            return V*pot
        return potential

    #I can give a list of positions to this functions to create vacancies at the specified positions.
    #I give it positions_vac
    def vacancy(positions, radius):
        @pb.site_state_modifier
        def modifier(state, x, y):
            length = len(positions)
            for i in range(length):
                pos = positions[i]
                state[(x-pos[0])**2 + (y-pos[1])**2 < radius**2] = False
            return state
        return modifier
    
    def rectangle(width, height):
        x0 = width / 2
        y0 = height / 2
        return pb.Polygon([[x0, y0], [x0, -y0], [-x0, -y0], [-x0, y0]])

    def circle(radius):
        def contains(x, y, z):
            return np.sqrt(x**2 + y**2) < radius
        return pb.FreeformShape(contains, width=[2*radius, 2*radius])

    #Here is the model
    #If there is no dopant or no vacancy, I don't put any
    if len(pos_dopants) == 0:
        model = pb.Model(
            monolayer_graphene(), 
            #pb.translational_symmetry(),
            #circle(radius = l_model),
            rectangle(l_model,l_model)
        )
    else:
         model = pb.Model(
            monolayer_graphene(), 
            #pb.translational_symmetry(),
            #circle(radius = R),
            rectangle(l_model,l_model),
            dopant(pos_dopants, sigma = sig, V = U),
            vacancy(pos_vac, radius=0.01),
            #constant_magnetic_field(B=50)
            #extra_hopping(pos_extra_hop, coupling = coupling)
        )
    #I plot the model
    '''
    plt.figure(figsize=(10, 10))
    model.plot()
    model.onsite_map.plot(cmap="Blues", site_radius=0.04)
    pb.pltutils.colorbar(label="U (eV)")
    plt.show()
    '''
    #The I solve it
    kpm = pb.kpm(model)
    #I add 1nm to l_STM to compute the LDOS (just in case; the model is already 3nm larger)
    l_LDOS = l_STM + 1
    energies = np.linspace(-E_range, E_range, E_reso)
    #Here I get the ldos
    spatial_ldos = kpm.calc_spatial_ldos(energy=energies, broadening=gamma,  # eV
                                        shape=rectangle(l_LDOS,l_LDOS))

    #I do this to fill some variables and not do it at every iteration in the loop
    smap = spatial_ldos.structure_map(0)
    ldos_data = np.array(smap.data,ndmin=1)
    ldos_positions = np.transpose(np.array(smap.positions))
    #print ("positions are:", ldos_positions)
    num_pts = ldos_positions.shape[0]
    ldos = np.zeros((energies.shape[0], num_pts, 4)) #an array where I put all the LDOS at all energies
    i=0
    for energy in energies:
        smap = spatial_ldos.structure_map(energy)
        ldos_data = np.array(smap.data,ndmin=1)
        ldos_positions = np.transpose(np.array(smap.positions))
        ldos[i,:,0:3] = ldos_positions[:,0:3]
        ldos[i,:,3] = ldos_data
        i +=1
    #print ("ldos is:",ldos)
    return ldos, model

def few_dopants_distri_MLG(l_STM, num_dopants, num_vac, theta, shift_x, shift_y):
    #This function places a few dopants randomly in the area delimited by l_STM
    a = 0.24595   # [nm] unit cell length
    a_cc = a/(sqrt(3))  # [nm] carbon-carbon distance

    #I define the basis vectors
    a1 = np.array([a, 0])
    a2 = np.array([a/2, a/2 * sqrt(3)])
    sub_B = np.array([0, a_cc])
    
    #I include the shift so that there is not always an atom in the center
    shift = np.array([shift_x, shift_y])

    #I rotate the lattice vectors by theta
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]])
    a1 = np.matmul(rot_matrix, a1)
    a2 = np.matmul(rot_matrix, a2)
    sub_B = np.matmul(rot_matrix, sub_B)

    #n_max is such that the diamond delimited by +/-n_max*a1+/-n_max*a2 englobes the STM image of size l_STM
    n_max = np.round(l_STM*2) 

    pos_dopants = [] #a list of the positions of all the dopants
    pos_vac = []
    for i in range(int(num_dopants)):
        m = np.random.randint(-n_max, high = n_max + 1)
        n = np.random.randint(-n_max, high = n_max + 1)
        sub_B_coef = np.random.randint(0, high = 1 + 1)
        pos_ = np.add(np.add(m*a1,n*a2),sub_B_coef*sub_B)
        #This condition is to check the dopant is in the image
        if ((pos_[0]>= -l_STM/2) and (pos_[0]<= l_STM/2) and (pos_[1]>= -l_STM/2) and (pos_[1]<= l_STM/2)):
            pos_dopants.append(np.add(np.add(np.add(m*a1,n*a2),sub_B_coef*sub_B), shift))
    
    for i in range(int(num_vac)):
        m = np.random.randint(-n_max, high = n_max + 1)
        n = np.random.randint(-n_max, high = n_max + 1)
        pos_ = np.add(np.add(m*a1,n*a2),0*sub_B)
        #This condition is to check the dopant is in the image
        if ((pos_[0]>= -l_STM/2) and (pos_[0]<= l_STM/2) and (pos_[1]>= -l_STM/2) and (pos_[1]<= l_STM/2)):
            pos_vac.append(np.add(np.add(np.add(m*a1,n*a2),0*sub_B), shift))

    return pos_dopants, pos_vac
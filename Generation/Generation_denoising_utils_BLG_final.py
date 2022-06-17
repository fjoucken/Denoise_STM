import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
import time
import io
import os.path
import matplotlib

from math import sqrt, pi
a = 0.24595   # [nm] unit cell length
a_cc = 0.142  # [nm] carbon-carbon distance

def ldos_from_params_BLG(pos_dopants, theta, shift_x, shift_y,  l_STM, U, sig, E_range, E_reso, gamma):
    #First I define the model
    l_model = l_STM + 15
    #l_model = l_STM + np.random.randint(2, high = 10 + 1)

   #I define the basis vectors
    a1 = np.array([a, 0, 0])
    a2 = np.array([a/2, a/2 * sqrt(3), 0])
    sub_A1 = np.array([0, a_cc, 0])
    sub_B1 = np.array([0, 0, 0])
    sub_A2 = np.array([0, 0, -0.335])
    sub_B2 = np.array([0, -a_cc, -0.335])

    #I rotate the lattice vectors by theta
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta),0],
             [np.sin(theta), np.cos(theta),0],
             [0, 0, 1]])
    a1 = np.matmul(rot_matrix, a1)
    a2 = np.matmul(rot_matrix, a2)
    sub_A1 = np.matmul(rot_matrix, sub_A1)
    sub_B1 = np.matmul(rot_matrix, sub_B1)
    sub_A2 = np.matmul(rot_matrix, sub_A2)
    sub_B2 = np.matmul(rot_matrix, sub_B2)
    shift = np.array([shift_x, shift_y, 0])

    def bilayer_graphene():
        t0 =3.16      # [eV] nearest neighbour hopping
        t1=0.42
        t3=0.38
        t4=0.14
        lat = pb.Lattice(a1,
                        a2)
        lat.add_sublattices(('A1', np.add(sub_A1, shift)),
                            ('B1', np.add(sub_B1, shift)),
                            ('A2', np.add(sub_A2, shift)),
                            ('B2', np.add(sub_B2, shift)))
        lat.add_hoppings(
            # inside the main cell
            ([0,  0], 'A1', 'B1', t0),
            ([0,  0], 'A2', 'B2', t0),
            ([0,  0], 'A2', 'B1', t1),
            # between neighboring cells
            ([1, -1], 'B1', 'A1', t0),
            ([0, -1], 'B1', 'A1', t0),
            ([1, -1], 'B2', 'A2', t0),
            ([0, -1], 'B2', 'A2', t0),
            #t3
            ([0,  -1], 'B2', 'A1', t3),
            ([1,  -1], 'B2', 'A1', t3),
            ([1,  -2], 'B2', 'A1', t3),
            #t4
            ([0,  0], 'A2', 'A1', t4),
            ([0,  -1], 'A2', 'A1', t4),
            ([1,  -1], 'A2', 'A1', t4),
        )
        return lat
    '''
    plt.rcParams['figure.figsize'] = [7, 7]
    lattice = bilayer_graphene()
    lattice.plot()
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()
    '''
    def gap(delta):
        """Break sublattice symmetry with opposite A and B onsite energy"""
        @pb.onsite_energy_modifier
        def potential(energy, sub_id):
            energy[sub_id == 'A1'] += delta/2
            energy[sub_id == 'B1'] += delta/2
            energy[sub_id == 'A2'] -= delta/2
            energy[sub_id == 'B2'] -= delta/2
            return energy
        return potential

    def circle(radius):
        def contains(x, y, z):
            return np.sqrt(x**2 + y**2) < radius
        return pb.FreeformShape(contains, width=[2*radius, 2*radius])

    def rectangle(width, height):
        x0 = width / 2
        y0 = height / 2
        return pb.Polygon([[x0, y0], [x0, -y0], [-x0, -y0], [-x0, y0]])

    def dopant(positions, sigma, V):
        @pb.onsite_energy_modifier
        def potential(x, y, z):
            length = len(positions)
            pot = 0
            for i in range(length):
                x0, y0, z0 = positions[i]
                pot += np.exp(-0.5*((x-x0)**2+(y-y0)**2+(z-z0)**2)/(sigma**2))
            return V*pot
        return potential
    
    def vacancy(positions, radius):
        @pb.site_state_modifier
        def modifier(state, x, y, z):
            length = len(positions)
            for i in range(length):
                pos = positions[i]
                state[(x-pos[0])**2 + (y-pos[1])**2 + (z-pos[2])**2 < radius**2] = False
            return state
        return modifier
    if len(pos_dopants) == 0:
        model = pb.Model(
            bilayer_graphene(), 
            #pb.translational_symmetry(),
            #circle(radius = R),
            rectangle(l_model,l_model)
        )
    else:
         model = pb.Model(
            bilayer_graphene(), 
            #pb.translational_symmetry(),
            #circle(radius = R),
            rectangle(l_model,l_model),
            dopant(pos_dopants, sigma = sig, V = U),
            #vacancy(pos_vac, radius=0.01),
            #constant_magnetic_field(B=50)
            #extra_hopping(pos_extra_hop, coupling = coupling)
        )
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
    return ldos

def few_dopants_distri_BLG(l_STM, num_dopants, num_vac, theta, shift_x, shift_y):
    #This function takes l_STM, c (the dopants concentration), and theta (the rotation of the lattice) and returns dopants distributed
    #randomly in the lattice 
    #It returns:
    #positions: a list of arrays with all the dopants positions
    #positions_vac
    #defects_pos: a list of arrays with all the defects positions (for pairs or 3N, the center of gravity of the pair)
    #defect_type: the corresponding list of defects types (it respects the order in the defects_pos list).

    a = 0.24595   # [nm] unit cell length
    a_cc = a/(sqrt(3))  # [nm] carbon-carbon distance

    #I define the basis vectors
    a1 = np.array([a, 0, 0])
    a2 = np.array([a/2, a/2 * sqrt(3), 0])
    sub_A1 = np.array([0, a_cc, 0])
    sub_B1 = np.array([0, 0, 0])
    sub_A2 = np.array([0, 0, -0.335])
    sub_B2 = np.array([0, -a_cc, -0.335])

    #I rotate the lattice vectors by theta
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta),0],
             [np.sin(theta), np.cos(theta),0],
             [0, 0, 1]])
    a1 = np.matmul(rot_matrix, a1)
    a2 = np.matmul(rot_matrix, a2)
    sub_A1 = np.matmul(rot_matrix, sub_A1)
    sub_B1 = np.matmul(rot_matrix, sub_B1)
    sub_A2 = np.matmul(rot_matrix, sub_A2)
    sub_B2 = np.matmul(rot_matrix, sub_B2)

    shift = np.array([shift_x, shift_y, 0])   

    #n_max is such that the diamond delimited by +/-n_max*a1+/-n_max*a2 englobes the STM image of size l_STM
    n_max = np.round(l_STM*2) 

    pos_dopants = [] #a list of the positions of all the dopants
    pos_vac = []

    #I put randomly a quarter of the dopants on each sublattices
    for i in range (num_dopants):
        site = np.random.randint(0, 4)
        #A1
        if site == 0:
            m = np.random.randint(-n_max, high = n_max + 1)
            n = np.random.randint(-n_max, high = n_max + 1)
            pos_ = np.add(np.add(m*a1,n*a2),0)
            if ((pos_[0]>= -l_STM/2) and (pos_[0]<= l_STM/2) and (pos_[1]>= -l_STM/2) and (pos_[1]<= l_STM/2)):
                pos_dopants.append(np.add(np.add(np.add(m*a1,n*a2),sub_A1), shift))
        #B1
        if site == 1:
            m = np.random.randint(-n_max, high = n_max + 1)
            n = np.random.randint(-n_max, high = n_max + 1)
            pos_ = np.add(np.add(m*a1,n*a2),0)
            if ((pos_[0]>= -l_STM/2) and (pos_[0]<= l_STM/2) and (pos_[1]>= -l_STM/2) and (pos_[1]<= l_STM/2)):
                pos_dopants.append(np.add(np.add(np.add(m*a1,n*a2),sub_B1), shift))
        #A2
        if site == 2:
            m = np.random.randint(-n_max, high = n_max + 1)
            n = np.random.randint(-n_max, high = n_max + 1)
            pos_ = np.add(np.add(m*a1,n*a2),0)
            if ((pos_[0]>= -l_STM/2) and (pos_[0]<= l_STM/2) and (pos_[1]>= -l_STM/2) and (pos_[1]<= l_STM/2)):
                pos_dopants.append(np.add(np.add(np.add(m*a1,n*a2),sub_A2), shift))
        #B2
        if site == 3:
            m = np.random.randint(-n_max, high = n_max + 1)
            n = np.random.randint(-n_max, high = n_max + 1)
            pos_ = np.add(np.add(m*a1,n*a2),0)
            if ((pos_[0]>= -l_STM/2) and (pos_[0]<= l_STM/2) and (pos_[1]>= -l_STM/2) and (pos_[1]<= l_STM/2)):
                pos_dopants.append(np.add(np.add(np.add(m*a1,n*a2),sub_B2), shift))

    #same for the vacancy
    for i in range (num_vac):
        site = np.random.randint(0, 4)
        #A1
        if site == 0:
            m = np.random.randint(-n_max, high = n_max + 1)
            n = np.random.randint(-n_max, high = n_max + 1)
            pos_ = np.add(np.add(m*a1,n*a2),0)
            if ((pos_[0]>= -l_STM/2) and (pos_[0]<= l_STM/2) and (pos_[1]>= -l_STM/2) and (pos_[1]<= l_STM/2)):
                pos_vac.append(np.add(np.add(np.add(m*a1,n*a2),sub_A1), shift))
        #B1
        if site == 1:
            m = np.random.randint(-n_max, high = n_max + 1)
            n = np.random.randint(-n_max, high = n_max + 1)
            pos_ = np.add(np.add(m*a1,n*a2),0)
            if ((pos_[0]>= -l_STM/2) and (pos_[0]<= l_STM/2) and (pos_[1]>= -l_STM/2) and (pos_[1]<= l_STM/2)):
                pos_vac.append(np.add(np.add(np.add(m*a1,n*a2),sub_B1), shift))
        #A2
        if site == 2:
            m = np.random.randint(-n_max, high = n_max + 1)
            n = np.random.randint(-n_max, high = n_max + 1)
            pos_ = np.add(np.add(m*a1,n*a2),0)
            if ((pos_[0]>= -l_STM/2) and (pos_[0]<= l_STM/2) and (pos_[1]>= -l_STM/2) and (pos_[1]<= l_STM/2)):
                pos_vac.append(np.add(np.add(np.add(m*a1,n*a2),sub_A2), shift))
        #B2
        if site == 3:
            m = np.random.randint(-n_max, high = n_max + 1)
            n = np.random.randint(-n_max, high = n_max + 1)
            pos_ = np.add(np.add(m*a1,n*a2),0)
            if ((pos_[0]>= -l_STM/2) and (pos_[0]<= l_STM/2) and (pos_[1]>= -l_STM/2) and (pos_[1]<= l_STM/2)):
                pos_vac.append(np.add(np.add(np.add(m*a1,n*a2),sub_B2), shift))

    return pos_dopants, pos_vac
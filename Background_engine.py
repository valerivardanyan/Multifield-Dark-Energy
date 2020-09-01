### *** This code solves the background equations of motion of arXiv:2008.13660 *** ###
### --- Background_engine.py performs the main calculations.
### --- model.py contains the definition of the model.
### --- Spinning_Quintessence.ipynb is the wrapper notebook.
### --- Plotting.ipynb produces the plots in the paper.

from model import *
import numpy as np

### ---> Returns a multidimensional matrix of Chrisoffel symbols, given a metric.
def Christoffels(phi_vec):

    metric_tmp = metric(phi_vec)[0]
    metric_der_tmp = metric(phi_vec)[1]

    coord_indices = [0, 1]
    Gamma_matrix = np.ones((len(coord_indices), len(coord_indices), len(coord_indices))) # Initialize the Christoffel matrix

    for i_indx in coord_indices:
        for k_indx in coord_indices:
            for l_indx in coord_indices:

                term_1 = term_2 = term_3 = 0.
                for m_indx in coord_indices:

                    inverse = np.linalg.inv(metric_tmp)[i_indx][m_indx]
                    term_1 = term_1 + inverse*metric_der_tmp[l_indx][m_indx][k_indx]
                    term_2 = term_2 + inverse*metric_der_tmp[k_indx][m_indx][l_indx]
                    term_3 = term_3 + inverse*metric_der_tmp[m_indx][k_indx][l_indx]

                Gamma_matrix[i_indx, k_indx, l_indx] = 0.5*(term_1 + term_2 - term_3)

    return Gamma_matrix

### ---> Returns the potential slow-roll parameter (Eq. 15)
def eps_V(phi_vec, params):

    phi1, phi2 = phi_vec

    V_der_tmp = pot_der(phi_vec, params) #G radients of the potential
    V_tmp = pot(phi_vec, params) # Get the potential

    metric_tmp = metric(phi_vec)[0] # Get the metric as a matrix. [1] are the component derivatives.
    inverse_tmp = np.linalg.inv(metric_tmp) # Inverse metric

    num_tmp = np.matmul(np.matmul(inverse_tmp, V_der_tmp), V_der_tmp)
    denom_tmp = 2.*V_tmp**2

    return num_tmp/denom_tmp

### ---> Returns $\Omega^2/9H^2$ (Eq. ), as well as the the potential slow-roll parameter (Eq. 15), this time calculated using V_N and V_T.
def Omega_term(phi_vec, velo_vec, N, params):

    phi1, phi2 = phi_vec
    phi1_prime, phi2_prime = velo_vec

    metric_tmp = metric(phi_vec)[0] # Get the metric as a matrix. [1] are the component derivatives.
    inverse_tmp = np.linalg.inv(metric_tmp) # Inverse metric
    pot_der_tmp = pot_der(phi_vec, params) # Gradients of the potential

    phi_prime_sqr = np.matmul(np.matmul(metric_tmp, velo_vec), velo_vec) # Square of \phi^\prime (Eq. 5 in number of e-foldings)
    tmp_hbl_sqr = Hubble_func_sqr(phi_vec, velo_vec, N, params) # Current value of H^2

    V_prime = np.dot(pot_der_tmp, velo_vec) # Time derivative of the potential

    Omega_term_1 = np.matmul(np.matmul(inverse_tmp, pot_der_tmp), pot_der_tmp)
    Omega_term_2 = V_prime**2/phi_prime_sqr
    Omega_sqr = (Omega_term_1 - Omega_term_2)/phi_prime_sqr/tmp_hbl_sqr # Can be derived from Eq. 12

    V_N_sqr = phi_prime_sqr*tmp_hbl_sqr*Omega_sqr # Eq. 10
    V_phi_sqr = V_prime**2/phi_prime_sqr # Follows from Eq. 6

    eps_potential = (V_N_sqr + V_phi_sqr)/2./pot(phi_vec, params)**2 # Eq. 15

    return [Omega_sqr/9./tmp_hbl_sqr, eps_potential]

### ---> Returns the Friedmann equation
def Hubble_func_sqr(phi_vec, velo_vec, N, params):
    Omega_r = params[-1]
    Omega_m = params[-2]

    metric_tmp = metric(phi_vec)[0]
    phi_prime_sqr = np.matmul(np.matmul(metric_tmp, velo_vec), velo_vec)

    matter_tmp = Omega_m*np.exp(-3.*N)*3. + Omega_r*np.exp(-4.*N)*3.
    tmp_num = pot(phi_vec, params) + matter_tmp
    tmp_denom = 3. - phi_prime_sqr/2.

    return tmp_num/tmp_denom

### ---> Returns the Hubble slow-roll parameter (Eq. 4, with everything normalized by H_0^2)
def eps(phi_vec, velo_vec, N, params):
    Omega_r = params[-1]
    Omega_m = params[-2]

    tmp_hbl_sqr = Hubble_func_sqr(phi_vec, velo_vec, N, params)

    metric_tmp = metric(phi_vec)[0]
    phi_prime_sqr = np.matmul(np.matmul(metric_tmp, velo_vec), velo_vec)

    num_tmp_1 = +1.*(phi_prime_sqr/2.)*tmp_hbl_sqr
    num_tmp_2 = -1.*pot(phi_vec, params)
    num_tmp_3 = 0.*Omega_m*np.exp(-3.*N)*3.
    num_tmp_4 = (1./3.)*Omega_r*np.exp(-4.*N)*3.

    eos = (num_tmp_1 + num_tmp_2 + num_tmp_3 + num_tmp_4)/3./tmp_hbl_sqr

    return 3.*(eos + 1.)/2.


### ---> Returns the r.h.s. of the scalar field equations of motion (Eq. 2 rewritten in number of e-folds)
def diff_evolve(y, N, params):
    Omega_r = params[-1]
    Omega_m = params[-2]
    phi1, phi1_prime, phi2, phi2_prime = y

    field_vector = np.array([phi1, phi2])
    velo_vector = np.array([phi1_prime, phi2_prime])

    metric_tmp = metric(field_vector)[0] # Get the metric as a matrix. [1] are the component derivatives.
    inverse_tmp = np.linalg.inv(metric_tmp)

    Gamma_tmp = Christoffels(field_vector)
    tmp_eps = eps(field_vector, velo_vector, N, params)
    tmp_hbl_sqr = Hubble_func_sqr(field_vector, velo_vector, N, params)
    pot_der_tmp = pot_der(field_vector, params)


    source_vector = velo_vector*(tmp_eps - 3.) - np.matmul(inverse_tmp, pot_der_tmp)/tmp_hbl_sqr -\
                    np.matmul(np.matmul(Gamma_tmp, velo_vector), velo_vector)

    dydt = [velo_vector[0], source_vector[0], \
            velo_vector[1], source_vector[1]]

    return dydt

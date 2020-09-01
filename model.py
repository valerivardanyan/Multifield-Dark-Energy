### *** This code solves the background equations of motion of arXiv:2008.13660 *** ###
### --- Background_engine.py performs the main calculations.
### --- model.py contains the definition of the model.
### --- Spinning_Quintessence.ipynb is the wrapper notebook.
### --- Plotting.ipynb produces the plots in the paper.

import numpy as np

### ---> Returns the two-field potential.
def pot(phi_vec, params):

    model = params[-3]
    phi1, phi2 = phi_vec

    if model == "tilted_higgs_like_exp":
        m, vev, V0, alpha, model, Omega_m, Omega_r = params

        power = 2.
        tmp_pot_1 = m**2*(phi1 - vev)**power/2.
        _lambda = 20.1
        tmp_pot_2 = alpha*np.exp(_lambda*phi2) + V0# - alpha*phi2

        return tmp_pot_1 + tmp_pot_2

    elif model == "tilted_higgs_like":
        m, vev, V0, alpha, model, Omega_m, Omega_r = params

        power = 2.
        tmp_pot_1 = m**2*(phi1 - vev)**power/2.
        tmp_pot_2 = V0 - alpha*phi2

        return tmp_pot_1 + tmp_pot_2
    else:
        print("Invalid model!")


### ---> Returns the derivative of the potential
def pot_der(phi_vec, params):

    phi1, phi2 = phi_vec
    model = params[-3]

    if model == "tilted_higgs_like_exp":
        m, vev, V0, alpha, model, Omega_m, Omega_r = params

        power = 2.
        pot_der_1 = power*m**2*(phi1 - vev)**(power - 1.)/2.
        _lambda = 20.1
        pot_der_2 = alpha*_lambda*np.exp(_lambda*phi2)# - alpha

        return np.array([pot_der_1, pot_der_2])

    elif model == "tilted_higgs_like":
        m, vev, V0, alpha, model, Omega_m, Omega_r = params

        power = 2.
        pot_der_1 = power*m**2*(phi1 - vev)**(power - 1.)/2.
        pot_der_2 = - alpha

        return np.array([pot_der_1, pot_der_2])

    else:
        print("Invalid model!")

### ---> Returns the metric and its derivatives
def metric(phi_vec):

    phi1, phi2 = phi_vec

    metric_tmp = np.array([[1., 0.], [0., phi1**2]]) # Here we hard-code the metric
    metric_der_phi1 = np.array([[0., 0.], [0., 2.*phi1]]) # Here we hard-code the phi_1 derivatives
    metric_der_phi2 = np.array([[0., 0.], [0., 0.]]) # Here we hard-code the phi_2 derivatives

    metric_der = np.array([metric_der_phi1, metric_der_phi2])

    if metric_tmp[0][1] != metric_tmp[1][0]:
        print("Non-symmetric metrics are not supported!")
    elif np.linalg.det(metric_tmp) == 0.:
        print("Singular metrics are not supported!")
    else:
        return [metric_tmp, metric_der]

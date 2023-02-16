'''
In this program we will integrate the equations of motion, and use them to plot the 
different parameters and the EoS parameter as functions of the redshift
'''
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.constants as const 

# Defining constants 
G = const.G				# Newton's gravitational constant [m^3 kg^-1 s^-2]
pi = const.pi 			
K_sqrd = 8 * pi * G		# Shorthand notation 
K = np.sqrt(K_sqrd)

# Defining the potentials
def V(phi, potential_type, alpha=None, chi=None, V_0=None):
	'''
	The potential can either be an 
	inverse power-law potential, or
	an exponential potential
	'''
	if potential_type == 'inverse':

		M = 1.

		if alpha == None:

			print('No value for alpha provided, exiting.')
			exit()

		else:

			potential = M**(4 + alpha) * phi**(-alpha)

			return potential 

	if potential_type == 'exponential':

		if chi == None or V_0 == None:

			print('No value for chi and/or V_0, exiting.')
			exit()

		else:

			potential = V_0 * np.exp(-K * chi * phi)

			return potential


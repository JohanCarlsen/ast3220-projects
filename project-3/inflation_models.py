import numpy as np 
import matplotlib.pyplot as plt 
from inflation import InflationModel

### Phi-squared potential functions ###
def phi2_pot(psi, psi_i):
	'''
	Dimensionless potential.
	'''
	v = 3 / (8 * np.pi) * (psi / psi_i)**2

	return v

def diff_phi2_pot(psi, psi_i):
	'''
	Differential of the phi2 potential
	'''
	dv = 3 / (4 * np.pi) * psi / psi_i**2

	return dv 

def phi2_epsilon(psi):
	'''
	SRA parameter epsilon
	'''
	return 1 / (4 * np.pi * psi**2)

### Starobinsky potential functions ###
def staro_pot(psi, psi_i):
	'''
	Starobinsky potential
	'''
	y = - np.sqrt(16 * np.pi / 3) * psi 
	y_i = - np.sqrt(16 * np.pi / 3) * psi_i

	v = 3 / (8 * np.pi) * ((1 - np.exp(y)) / (1 - np.exp(y_i)))**2

	return v 

def diff_staro_pot(psi, psi_i):
	'''
	Differential of the Starobinksy potential.
	'''
	y = - np.sqrt(16 * np.pi / 3) * psi 
	y_i = - np.sqrt(16 * np.pi / 3) * psi_i

	dv = np.sqrt(3 / np.pi) * (1 - np.exp(y)) * np.exp(y) / (1 - np.exp(y_i))**2

	return dv

def staro_epsilon(psi):
	'''
	SRA parameter epsilon
	'''
	y = - np.sqrt(16 * np.pi / 3) * psi 
	eps = 4 / 3 * np.exp(2 * y) / (1 - np.exp(y))**2

	return eps

def staro_eta(psi):
	'''
	SRA parameter eta
	'''
	y = - np.sqrt(16 * np.pi / 3) * psi 
	eta = 4 / 3 * (2 * np.exp(2 * y) - np.exp(y)) / (1 - np.exp(y))**2

	return eta


phi2_psi_i = 8.9251
phi2_tau_end = 4 * np.pi * phi2_psi_i**2 - 2 * np.sqrt(np.pi) / phi2_psi_i

phi2 = InflationModel(phi2_psi_i, 'phi2')
phi2.set_end_of_inflation_time(phi2_tau_end)
phi2.set_potential(phi2_pot)
phi2.set_potential_differential(diff_phi2_pot)
phi2.solve_scalar_field(1500)
phi2.plot_solutions(compare_to_SRA=True)
phi2.plot_solutions()
phi2.set_epsilon_parameter(phi2_epsilon)
phi2.num_e_folds()
phi2.pressure_energy_density_ratio()
phi2.epsilon_remainding_e_folds()
phi2.plot_tensor_to_scalar_ratio()

starobinsky = InflationModel(2, 'starobinsky')
starobinsky.set_end_of_inflation_time(2694.91)
starobinsky.set_potential(staro_pot)
starobinsky.set_potential_differential(diff_staro_pot)
starobinsky.solve_scalar_field(3000)
starobinsky.plot_solutions()
starobinsky.plot_solutions(xlim=[2680, 2750])
starobinsky.set_epsilon_parameter(staro_epsilon, print_tau_end=True)
starobinsky.set_eta_parameter(staro_eta)
starobinsky.num_e_folds(guess=None, text_ypos=0.6, plot=False)
starobinsky.pressure_energy_density_ratio(zoomed=True)#xlim=[2685, 2705])
starobinsky.epsilon_remainding_e_folds()
starobinsky.plot_tensor_to_scalar_ratio()

plt.show()
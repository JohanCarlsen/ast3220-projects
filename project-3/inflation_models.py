import numpy as np
import matplotlib.pyplot as plt
from inflation import InflationModel

### Phi-squared potential functions ###
def phi2_pot(psi, psi_i):
	'''
	Phi^2 potential.

	:param psi:		Scalar field, float or array
	:param psi_i:	Initial value of the scalar field, float
	:return:		Potential, float or array
	'''
	v = 3 / (8 * np.pi) * (psi / psi_i)**2

	return v

def diff_phi2_pot(psi, psi_i):
	'''
	Differential of the phi^2 potential.

	:param psi:		Scalar field, float or array
	:param psi_i:	Initial value for the scalar field, float
	:return:		Potential differential, float or array
	'''
	dv = 3 / (4 * np.pi) * psi / psi_i**2

	return dv

def phi2_epsilon(psi):
	'''
	SRA parameter epsilon for the phi^2 potential.

	:param psi:		Scalar field, float or array
	:return:		Epsilon, float or array
	'''
	return 1 / (4 * np.pi * psi**2)

### Starobinsky potential functions ###
def staro_pot(psi, psi_i):
	'''
	Starobinsky potential.

	:param psi:		Scalar field, float or array
	:param psi_i:	Initial value for the scalar field, float
	:return:		Potential, float or array
	'''
	# Shorthand notations
	y = - np.sqrt(16 * np.pi / 3) * psi
	y_i = - np.sqrt(16 * np.pi / 3) * psi_i

	v = 3 / (8 * np.pi) * ((1 - np.exp(y)) / (1 - np.exp(y_i)))**2

	return v

def diff_staro_pot(psi, psi_i):
	'''
	Differential of the Starobinksy potential.

	:param psi:		Scalar field, float or array
	:param psi_i:	Initial value for the scalar field, float
	:return:		Potential differential, float or array
	'''
	# Shorthand notations
	y = - np.sqrt(16 * np.pi / 3) * psi
	y_i = - np.sqrt(16 * np.pi / 3) * psi_i

	dv = np.sqrt(3 / np.pi) * (1 - np.exp(y)) * np.exp(y) / (1 - np.exp(y_i))**2

	return dv

def staro_epsilon(psi):
	'''
	SRA parameter epsilon for the Starobinsky potential.

	:param psi: Scalar field, float or array
	:return:	Epsilon, float or array
	'''
	# Shorthand notation
	y = - np.sqrt(16 * np.pi / 3) * psi
	eps = 4 / 3 * np.exp(2 * y) / (1 - np.exp(y))**2

	return eps

def staro_eta(psi):
	'''
	SRA parameter eta for the Starobinsky potential.

	:param psi:	Scalar field, float or array
	:return:	Eta, float or array
	'''
	# Shorthand notation
	y = - np.sqrt(16 * np.pi / 3) * psi
	eta = 4 / 3 * (2 * np.exp(2 * y) - np.exp(y)) / (1 - np.exp(y))**2

	return eta

# Initial value for the scalar field, found from the SRA.
phi2_psi_i = 8.9251

phi2 = InflationModel(phi2_psi_i, 'phi2')
phi2.set_potential(phi2_pot)
phi2.set_potential_differential(diff_phi2_pot)
phi2.solve_scalar_field(1500)
phi2.set_epsilon_parameter(phi2_epsilon, print_tau_end=True)
phi2.plot_solutions(compare_to_SRA=True)
phi2.plot_solutions()
phi2.num_e_folds()
phi2.pressure_energy_density_ratio()
phi2.SRA_params_remaining_e_folds()
phi2.plot_tensor_to_scalar_ratio()

# Initial value for the scalar field is set to 2.
staro_psi_i = 2

starobinsky = InflationModel(staro_psi_i, 'starobinsky')
starobinsky.set_potential(staro_pot)
starobinsky.set_potential_differential(diff_staro_pot)
starobinsky.solve_scalar_field(3000)
starobinsky.set_epsilon_parameter(staro_epsilon, print_tau_end=True)
starobinsky.set_eta_parameter(staro_eta)
starobinsky.plot_solutions()
starobinsky.num_e_folds(guess=None, plot=False)
starobinsky.pressure_energy_density_ratio(zoomed=True)
starobinsky.SRA_params_remaining_e_folds()
starobinsky.SRA_params_remaining_e_folds(include_approx=True)
starobinsky.plot_tensor_to_scalar_ratio()
starobinsky.plot_tensor_to_scalar_ratio(include_approx=True)

plt.show()

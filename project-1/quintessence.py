

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp, cumulative_trapezoid, simpson
from scipy.interpolate import interp1d

class Quintessence:

	def __init__(self, potential_type, dN):

		self.pot = potential_type
		self.dN = dN

############################################################################

	def set_redshift_range(self, z_max, z_min=0):

		N_min = -np.log(1 + z_max)
		N_max = -np.log(1 + z_min)
		N_steps = int(np.ceil((N_max - N_min) / self.dN))
		N = np.linspace(N_min, N_max, N_steps)

		self.N_range = [N_min, N_max]
		self.N = np.linspace(N_min, N_max, N_steps)
		self.z = np.exp(-N) - 1

############################################################################

	def power_law(self, N, variables):

		x1, x2, x3, lmbda = variables

		dx1 = -3 * x1 + np.sqrt(6) / 2 * lmbda * x2**2 + .5 * x1 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
		dx2 = -np.sqrt(6) / 2 * lmbda * x1 * x2 + .5 * x2 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
		dx3 = -2 * x3 + .5 * x3 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
		dlmbda = -np.sqrt(6) * lmbda**2 * x1 

		diffs = np.array([dx1, dx2, dx3, dlmbda])

		return diffs

############################################################################

	def exponential(self, N, variables):

		lmbda = 3 / 2
		x1, x2, x3 = variables

		dx1 = -3 * x1 + np.sqrt(6) / 2 * lmbda * x2**2 + .5 * x1 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
		dx2 = -np.sqrt(6) / 2 * lmbda * x1 * x2 + .5 * x2 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
		dx3 = -2 * x3 + .5 * x3 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)

		diffs = np.array([dx1, dx2, dx3])

		return diffs

############################################################################

	def integrate(self):

		if self.pot == 'power-law':

			x1_0 = 5e-5
			x2_0 = 1e-8
			x3_0 = 0.9999
			lmbda_0 = 1e9

			initial_variables = np.array([x1_0, x2_0, x3_0, lmbda_0])

			sol = solve_ivp(self.power_law, self.N_range, initial_variables, method='RK45', rtol=1e-9, atol=1e-9, dense_output=True)

		if self.pot == 'exponential':

			x1_0 = 0.
			x2_0 = 5e-13
			x3_0 = 0.9999

			initial_variables = np.array([x1_0, x2_0, x3_0])

			sol = solve_ivp(self.exponential, self.N_range, initial_variables, method='RK45', rtol=1e-9, atol=1e-9, dense_output=True)

		variables = sol.sol(self.N)

		# Extracting the solutions from solve_ivp
		x1 = variables[0, :]
		x2 = variables[1, :]
		x3 = variables[2, :]

		# Calculating the parameters
		self.O_m = 1 - x1**2 - x2**2 - x3**2
		self.O_phi = x1**2 + x2**2
		self.O_r = x3**2

		# Calculating the EoS parameter
		self.w_phi = (x1**2 - x2**2) / (x1**2 + x2**2)

############################################################################

	def Hubble_parameter(self):

		integrand = 3 * (1 + np.flip(self.w_phi))
		I = cumulative_trapezoid(integrand, self.N, initial=0)
		self.H = np.sqrt(self.O_m[-1] * np.exp(-3 * self.N) + self.O_r[-1] * np.exp(-4 * self.N) + self.O_phi[-1] * np.exp(np.flip(I)))

		return self.H

############################################################################

	def universe_age(self):

		integrand = 1 / self.H 
		self.t_0 = simpson(integrand, self.N) 

		return self.t_0 

############################################################################

	def luminosity_distance(self, z_max=2):

		ln3_search = np.logical_and(self.N <= -np.log(z_max + 1) + self.dN, self.N >= -np.log(z_max + 1) - self.dN)
		idx = np.where(ln3_search == True)[0][0]
		N_ln3 = self.N[idx:]
		H = self.H[idx:]
		
		integrand = np.exp(-N_ln3) / H
		I = cumulative_trapezoid(integrand, N_ln3, initial=0)

		self.dL = np.exp(-N_ln3) * np.flip(I) 

		return self.z[idx:], self.dL

############################################################################

	def plot_parameters(self):

		if self.pot == 'power-law':

			title = r'$V(\phi)=M^{4+\alpha}\phi^{-\alpha}$'

		if self.pot == 'exponential':

			title = r'$V(\phi)=V_0e^{-\kappa\zeta\phi}$'

		plt.plot(self.z, self.O_m, ls='dashed', color='black', label=r'$\Omega_m$')
		plt.plot(self.z, self.O_r, ls='dotted', color='black', label=r'$\Omega_r$')
		plt.plot(self.z, self.O_phi, ls='dashdot', color='black', label=r'$\Omega_\phi$')
		plt.plot(self.z, self.w_phi, color='black', label=r'$w_\phi$')

		plt.title(title)
		plt.xlabel(r'Redshift $z$')
		plt.xscale('log')
		plt.legend()

############################################################################

	def plot_Hubble_parameter(self):

		if self.pot == 'power-law':

			lab = r'$H^{power-law}$'
			style= (0, (5, 10))

		if self.pot == 'exponential':

			lab = r'$H^{exponential}$'
			style = 'dotted'

		H = self.Hubble_parameter()

		plt.plot(self.z, H, ls=style, color='black', label=lab)

		plt.xlabel(r'Redshift $z$')
		plt.ylabel(r'$H/H_0$')
		plt.xscale('log')
		plt.yscale('log')
		plt.legend()

############################################################################

	def plot_luminosity_distance(self):

		if self.pot == 'power-law':

			lab = r'$d_L^{power-law}$'
			style= (0, (5, 10))

		if self.pot == 'exponential':

			lab = r'$d_L^{exponential}$'
			style = 'dotted'

		z, dL = self.luminosity_distance()

		plt.plot(z, dL, ls=style, color='black', label=lab)

		plt.xlabel(r'Redshift $z$')
		plt.ylabel(r'$H_0d_L/c$')
		plt.legend()

############################################################################
############################################################################

if __name__ == '__main__':

	'''
	We have to have some functions that can compute values for the LCDM model, 
	as it will not be an instance of the above class. 
	'''
	def luminosity_distance(Hubble_parameter):
		'''
		Luminosity distance for the LCDM model.
		'''
		ln3_search = np.logical_and(N <= -np.log(3) + dN, N >= -np.log(3) - dN)
		idx = np.where(ln3_search == True)[0][0]
		ln3 = N[idx]
		N_ln3 = N[idx:]
		H = Hubble_parameter[idx:]
		
		integrand = np.exp(-N_ln3) / H
		I = cumulative_trapezoid(integrand, N_ln3, initial=0)

		dL = np.exp(-N_ln3) * np.flip(I) 

		return z[idx:], dL


	def age_of_Universe(Hubble_parameter):
		'''
		Age of the Universe for the LCDM model.
		'''
		H = Hubble_parameter
		integrand = 1 / H
		t_0 = simpson(integrand, N)

		return t_0


	def chi_squared(computed_model, data):
		'''
		Perform the chi squared fitting to the models.
		'''
		inter = interp1d(z_ln3, computed_model)		# Interpolating the data 
		model = inter(z_data) * 3 / .7 				# Converting to Mpc

		chi = np.sum((model - data)**2 / error_data**2)

		return chi

	'''
	Setting default plotting parameters.
	'''
	plt.rcParams['lines.linewidth'] = 1
	plt.rcParams['figure.figsize'] = (10, 3.5)

	z_max = 2e7
	dN = 1e-4

	expon = Quintessence('exponential', dN)
	power = Quintessence('power-law', dN)

	expon.set_redshift_range(z_max)
	power.set_redshift_range(z_max)

	expon.integrate()
	power.integrate()

	plt.figure()
	expon.plot_parameters()
	plt.tight_layout()
	plt.savefig('figures/exp_params.pdf')
	plt.savefig('figures/exp_params.png')

	plt.figure()
	power.plot_parameters()
	plt.tight_layout()
	plt.savefig('figures/pow_params.pdf')
	plt.savefig('figures/pow_params.png')

	N = expon.N 
	z = expon.z
	O_m0 = .3 		# Setting the Omega_m0 to 0.3 for the LCDM model

	H_CDM = np.sqrt(O_m0 * np.exp(-3 * N) + (1 - O_m0))
	z_crop, dL_CDM = luminosity_distance(H_CDM)

	plt.figure()
	expon.plot_Hubble_parameter()
	power.plot_Hubble_parameter()
	plt.plot(z, H_CDM, color='black', ls='dashed', label=r'$H^{\Lambda CDM}$')
	plt.legend()
	plt.tight_layout()
	plt.savefig('figures/hubble_param.pdf')
	plt.savefig('figures/hubble_param.png')

	plt.figure()
	expon.plot_luminosity_distance()
	power.plot_luminosity_distance()
	plt.plot(z_crop, dL_CDM, color='black', label=r'$dL^{\Lambda CDM}$')
	plt.legend()
	plt.tight_layout()
	plt.savefig('figures/lum_dist.pdf')
	plt.savefig('figures/lum_dist.png')

	t0_expon = f'{expon.universe_age():.4e}'
	t0_power = f'{power.universe_age():.4e}'
	t0_CDM = f'{age_of_Universe(H_CDM):.4e}'

	print(f"\n{'':<10} {'Power-law':<15} {'Exponential':<15} {'LCDM'}")
	print(f"{'H_0t_0':<10} {t0_power:<15} {t0_expon:<15} {t0_CDM}\n")

	z_data, dL_data, error_data = np.loadtxt('sndata.txt', skiprows=5, unpack=True)

	z_ln3, dL_power = power.luminosity_distance()
	_, dL_expon = expon.luminosity_distance()

	chi_expon = chi_squared(expon.dL, dL_data)
	chi_power = chi_squared(power.dL, dL_data)

	print(f'Chi-squared for exponential potential:\t{chi_expon:.4f}')
	print(f'Chi-squared for power-law potential:\t{chi_power:.4f}\n')

	O_m0_list = np.linspace(0, 1, 1001)
	chi_min = 500
	O_m0_best = 0

	for i in range(len(O_m0_list)):

		H = np.sqrt(O_m0_list[i] * np.exp(-3 * N) + (1 - O_m0_list[i]))
		_, dL = luminosity_distance(H)

		chi_result = chi_squared(dL, dL_data)

		if chi_result < chi_min:

			chi_min = chi_result
			O_m0_best = O_m0_list[i]

	print(f'Best fit for Omega_m0 for the LCDM model: {O_m0_best}\n')

	H_CDM_bestfit = np.sqrt(O_m0_best * np.exp(-3 * N) + (1 - O_m0_best))	# Best fit Hubble parameter
	__, dL_CDM_bestfit = luminosity_distance(H_CDM_bestfit)					# Best fit dL

	chi_LCDM_bestfit = chi_squared(dL_CDM_bestfit, dL_data)
	print(f'Chi squared for best fitted LCDM:\t{chi_LCDM_bestfit:.4f}\n')

	plt.show()

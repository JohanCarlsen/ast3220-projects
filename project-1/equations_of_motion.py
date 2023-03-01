'''
In this program we will integrate the equations of motion, and use them to plot the 
different parameters and the EoS parameter as functions of the redshift.
'''
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp, cumulative_trapezoid, simpson

'''
Settings for plots
'''
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.size'] = 12

def differentials_powerlaw(N, variables):
	'''
	This function returns the differentials of 
	x_i, to be used by solve_ivp. Variables
	are x1, x2, x3, and lambda.
	'''
	x1, x2, x3, lmbda = variables

	dx1 = -3 * x1 + np.sqrt(6) / 2 * lmbda * x2**2 + .5 * x1 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
	dx2 = -np.sqrt(6) / 2 * lmbda * x1 * x2 + .5 * x2 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
	dx3 = -2 * x3 + .5 * x3 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
	dlmbda = -np.sqrt(6) * lmbda**2 * x1 

	diffs = np.array([dx1, dx2, dx3, dlmbda])

	return diffs


def differentials_exponential(N, variables):
	'''
	Same as for the power law, except that lambda is constant.
	'''
	lmbda = 3 / 2
	x1, x2, x3 = variables

	dx1 = -3 * x1 + np.sqrt(6) / 2 * lmbda * x2**2 + .5 * x1 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
	dx2 = -np.sqrt(6) / 2 * lmbda * x1 * x2 + .5 * x2 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
	dx3 = -2 * x3 + .5 * x3 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)

	diffs = np.array([dx1, dx2, dx3])

	return diffs


def integrate(potential_type, dN):
	'''
	Function to integrate the system. The argument potential_type has
	to be either 'power-law' or 'exponential'.
	'''

	if potential_type == 'power-law':

		# Setting initial values at z = 2e7, ie. N = -ln(1 + 2e7)
		x1_0 = 5e-5
		x2_0 = 1e-8
		x3_0 = 0.9999
		lmbda_0 = 1e9

		initial_variables = np.array([x1_0, x2_0, x3_0, lmbda_0])

		sol = solve_ivp(differentials_powerlaw, N_range, initial_variables, method='RK45', rtol=1e-9, atol=1e-9, dense_output=True)
		
	if potential_type == 'exponential':

		# Initial values for exponential potential
		x1_0 = 0.
		x2_0 = 5e-13
		x3_0 = 0.9999

		initial_variables = np.array([x1_0, x2_0, x3_0])

		sol = solve_ivp(differentials_exponential, N_range, initial_variables, method='RK45', rtol=1e-9, atol=1e-9, dense_output=True)

	variables = sol.sol(N)

	# Extracting the solutions from solve_ivp
	x1 = variables[0, :]
	x2 = variables[1, :]
	x3 = variables[2, :]

	# Saving x1, x2, and x3 to be used later
	np.save('x1', x1)
	np.save('x2', x2)
	np.save('x3', x3)

	# Calculating the parameters
	Omega_m = 1 - x1**2 - x2**2 - x3**2
	Omega_phi = x1**2 + x2**2
	Omega_r = x3**2

	# Calculating the EoS parameter
	w_phi = (x1**2 - x2**2) / (x1**2 + x2**2)

	return Omega_m, Omega_phi, Omega_r, w_phi


def Hubble(Omega_m_0, Omega_r_0, Omega_phi_0, EoS_parameter):

	w_phi = EoS_parameter
	integrand = 3 * (1 + np.flip(w_phi))
	I = cumulative_trapezoid(integrand, N, initial=0)
	# I = np.trapz(integrand, N)

	H = np.sqrt(Omega_m_0 * np.exp(-3 * N) + Omega_r_0 * np.exp(-4 * N) + Omega_phi_0 * np.exp(np.flip(I)))

	return H


def age_of_Universe(Hubble_parameter):

	H = Hubble_parameter
	integrand = 1 / H
	t_0 = simpson(integrand, N)

	return t_0


def luminosity_distance(Hubble_parameter):

	ln3_search = np.logical_and(N <= -np.log(3) + dN, N >= -np.log(3) - dN)
	idx = np.where(ln3_search == True)[0][0]
	ln3 = N[idx]
	N_ln3 = N[idx:]
	H = Hubble_parameter[idx:]
	
	integrand = np.exp(-N_ln3) / H
	I = cumulative_trapezoid(integrand, N_ln3, initial=0)

	dL = np.exp(-N_ln3) * np.flip(I) 

	return z[idx:], dL


# The interval [0, 2e7] in z corresponds to [-ln(1 + 2e7), 0] in N
N_min = -np.log(1 + 2e7)
N_max = 0.
N_range = [N_min, N_max]

dN = 1e-4
N_steps = int(np.ceil((N_max - N_min) / dN))
N = np.linspace(N_min, N_max, N_steps)

# Converting from N to redshift (z)
z = np.exp(-N) - 1

Omega_m = np.zeros((2, N_steps))
Omega_phi = np.zeros((2, N_steps))
Omega_r = np.zeros((2, N_steps))
w_phi = np.zeros((2, N_steps))

Omega_m[0, :], Omega_phi[0, :], Omega_r[0, :], w_phi[0, :] = integrate('power-law', dN)
Omega_m[1, :], Omega_phi[1, :], Omega_r[1, :], w_phi[1, :] = integrate('exponential', dN)

# Setting some variables to loop over when plotting
plot_text = [r'$V(\phi)=M^5\phi^{-1}$', r'$V(\phi)=V_0e^{-\frac{3\kappa}{2}\phi}$']
text_pos = [(1e-3, 1.5), (1e-3, .5e126)]

# fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# for i in range(2):

# 	m = ax[i].plot(z, Omega_m[i, :], ls='dashed', color='black')
# 	r = ax[i].plot(z, Omega_r[i, :], ls='dotted', color='black')
# 	phi = ax[i].plot(z, Omega_phi[i, :], ls='dashdot', color='black')
# 	w = ax[i].plot(z, w_phi[i, :], color='black')

# 	ax[i].set_xlabel(r'$z$')
# 	ax[i].set_xscale('log')

# 	ax[i].text(1e-3, -.5, plot_text[i], fontsize=10)

# # Setting labels
# labels = [r'$\Omega_m$', r'$\Omega_r$', r'$\Omega_\phi$', r'$w_\phi$']
# fig.legend([m, r, phi, w], labels=labels, loc='right')

# plt.tight_layout()
# plt.savefig('figures/equations_of_motion.pdf')
# plt.savefig('figures/equations_of_motion.png')

Omega_m_0_CDM = .3

H_power = Hubble(Omega_m[0, -1], Omega_r[0, -1], Omega_phi[0, -1], w_phi[0, :])
H_exp = Hubble(Omega_m[1, -1], Omega_r[1, -1], Omega_phi[1, -1], w_phi[1, :])
H_CDM = np.sqrt(Omega_m_0_CDM * np.exp(-3 * N) + (1 - Omega_m_0_CDM))

# plt.figure(figsize=(10, 5))

# plt.plot(z, H_power, ls=(0, (5, 10)), color='black', label=r'$H(z)^{PL}$')
# plt.plot(z, H_exp, ls='dotted', color='black', label=r'$H(z)^{EXP}$')
# plt.plot(z, H_CDM, ls='dashed', color='black', label=r'$H(z)^{\Lambda CDM}$')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('z')
# plt.ylabel(r'$H(z)/H_0$')
# plt.legend()
# plt.savefig('Hubble-parameter.pdf')
# plt.savefig('Hubble-parameter.png')

# plt.show()

'''
Computing the age of the Universe in the different models 
'''
t_0_power = f'{age_of_Universe(H_power):.4e}'
t_0_exp = f'{age_of_Universe(H_exp):.4e}'
t_0_CDM = f'{age_of_Universe(H_CDM):.4e}'

print(f"\n{'':<10} {'Power-law':<15} {'Exponential':<15} {'LCDM'}")
print(f"{'H_0t_0':<10} {t_0_power:<15} {t_0_exp:<15} {t_0_CDM}\n")

'''
Calculating the luminosity distance for the power-law
potential model and the exponential potential model.
'''

z_ln3, dL_power = luminosity_distance(H_power)
_, dL_exp = luminosity_distance(H_exp)

plt.figure(figsize=(10, 5))
plt.plot(z_ln3, dL_power, ls='dashed', color='black', label=r'$d_L^{PL}$')
plt.plot(z_ln3, dL_exp, ls='dotted', color='black', label=r'$d_L^{EXP}$')

'''
Extracting the data from the file sndata.txt
'''
z_data, dL_data, error_data = np.loadtxt('sndata.txt', skiprows=5, unpack=True)

# plt.errorbar(z_data, dL_data, error_data, capsize=3, color='black')

plt.legend()
plt.xlabel('z')
plt.ylabel(r'$H_0d_L/c$')
plt.savefig('lum-dist.pdf')
plt.savefig('lum-dist.png')

plt.show()













	
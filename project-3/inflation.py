import numpy as np 
import matplotlib.pyplot as plt 
import scipy.constants as const
from scipy.integrate import simpson
from cycler import cycler

plt.rcParams.update({
	'text.usetex': True,
	'font.family': 'Helvetica',
	'font.size': 12,
	'figure.figsize': (8, 4.5),
	'xtick.direction': 'inout',
	'ytick.direction': 'inout',
	'lines.linewidth': 1,
	'axes.prop_cycle': cycler(color=['k', 'r', 'royalblue'])
	})

class InflationModel:

	def __init__(self, scalar_field_init_value, model_name):

		self.psi_i = scalar_field_init_value
		self.name = str(model_name) + '_'
		self.tau_end = None
		self.eta = None

	def set_end_of_inflation_time(self, end_time):

		self.tau_end = end_time

	def set_potential(self, potential):

		self.potential = potential 

	def set_potential_differential(self, pot_diff):

		self.pot_diff = pot_diff

	def double_time_derivative_scalar_field(self, psi, dpsi, h):

		dv = self.pot_diff(psi, self.psi_i)

		return - 3 * h * dpsi - dv

	def dimensionless_Hubble_parameter(self, psi, dpsi):

		v = self.potential(psi, self.psi_i)

		return np.sqrt(8 * np.pi / 3 * (1/2 * dpsi**2 + v))

	def solve_scalar_field(self, total_time, dtau=1e-2):

		N = int(np.ceil(total_time / dtau))
		tau = np.linspace(0, total_time, N+1)

		h = np.zeros(N+1)
		psi = np.zeros(N+1)
		dpsi = np.zeros(N+1)
		ln_a_ai = np.zeros(N+1)

		h[0] = 1; psi[0] = self.psi_i

		for i in range(N):

			d2psi = self.double_time_derivative_scalar_field(psi[i], dpsi[i], h[i])

			ln_a_ai[i+1] = ln_a_ai[i] + h[i] * dtau
			dpsi[i+1] = dpsi[i] + d2psi * dtau
			psi[i+1] = psi[i] + dpsi[i+1] * dtau
			h[i+1] = self.dimensionless_Hubble_parameter(psi[i+1], dpsi[i+1])

		self.h = h; self.psi = psi; self.h2 = h**2
		self.dpsi = dpsi; self.ln_a_ai = ln_a_ai; self.tau = tau

		self.psi_SRA = self.psi_i - 1 / (4 * np.pi * self.psi_i) * tau  
		self.h2_SRA = (1 - 1 / (4 * np.pi * self.psi_i**2) * tau)**2
		self.ln_a_ai_SRA = tau - 1 / (8 * np.pi * self.psi_i**2) * tau**2

	def plot_solutions(self, compare_to_SRA=False, tight_layout=False, xlim=None):

		filename = 'num-sols' if not compare_to_SRA else 'SRA-compare'

		tau, tau_end, psi, ln_a_ai, h2 = self.tau, self.tau_end, self.psi, self.ln_a_ai, self.h2

		if xlim is not None:

			min_idx = np.where(tau >= xlim[0])[0][0]
			max_idx = np.where(tau <= xlim[1])[0][-1]

			tau = tau[min_idx:max_idx]; h2 = h2[min_idx:max_idx]
			ln_a_ai = ln_a_ai[min_idx:max_idx]; psi = psi[min_idx:max_idx]

		with plt.rc_context({'figure.figsize': (10.6, 6)}):

			fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

			ax1.plot(tau, psi/self.psi_i, label='Exact')
			ax1.set_ylabel(r'$\psi/\psi_i$')

			ax2.plot(tau, ln_a_ai)
			ax2.set_ylabel(r'$\ln\left(\frac{a}{a_i}\right)$')

			ax3.plot(tau, h2)
			ax3.set_ylabel(r'$h^2$')
			ax3.set_xlabel(r'$\tau$')

			if compare_to_SRA:

				psi_SRA, ln_a_ai_SRA, h2_SRA = self.psi_SRA, self.ln_a_ai_SRA, self.h2_SRA

				ax1.plot(tau, psi_SRA / self.psi_i, ls='dashed', label='SRA')
				ax1.set_ylim([- 0.75, 0.65])
				ax1.legend()

				ax2.plot(tau, ln_a_ai_SRA, ls='dashed', label='SRA')				
				ax2.set_ylim([345, 655])

				ax3.plot(tau, h2_SRA, ls='dashed', label='SRA')				
				ax3.set_ylim([- 0.35, 0.45])
				ax3.set_xlim([850, 1510])

				if self.tau_end is not None:

					ax1.plot([tau_end, tau_end], [- 0.75, 0.65], ls='dotted', color='black')
					ax2.plot([tau_end, tau_end], [345, 655], ls='dotted', color='black')
					ax3.plot([tau_end, tau_end], [- 0.35, 0.45], ls='dotted', color='black')

			elif self.tau_end is not None:

				ax1.plot([tau_end, tau_end], [- 0.1, 1.1], ls='dotted', color='black')
				# ax1.set_ylim([- 0.1, 1.1])

				ax2.plot([tau_end, tau_end], [- 5, np.max(self.ln_a_ai) + np.max(self.ln_a_ai) * 0.05], ls='dotted', color='black')
				# ax2.set_ylim([- 5, np.max(self.ln_a_ai) + np.max(self.ln_a_ai) * 0.05])

				ax3.plot([tau_end, tau_end], [- 0.1, 1.1], ls='dotted', color='black')
				# ax3.set_ylim([- 0.1, 1.1])

			if tight_layout:

				fig.tight_layout()

			else:

				fig.subplots_adjust(hspace=0.05)

			# ax3.set_xlim(xlim)

			if xlim is not None:

				filename += '_zoomed'

			fig.savefig('figures/' + self.name + filename + '.pdf', bbox_inches='tight')
			fig.savefig('figures/' + self.name + filename + '.png', bbox_inches='tight')

	def set_epsilon_parameter(self, epsilon, print_tau_end=False):

		self.epsilon = epsilon(self.psi)
		end_idx = np.where(self.epsilon <= 1)[0][-1]
		self.tau_end = self.tau[end_idx]

		if print_tau_end:

			print(f'\nTime when epsilon <= 1 for the last time: {self.tau_end}')

	def set_eta_parameter(self, eta):

		self.eta = eta(self.psi)

	def num_e_folds(self, text_xpos=0, text_ypos=0.8, guess=500, plot=True):

		end_of_inflation_idx = np.where(self.epsilon <= 1)[0][-1]

		self.N_tot = self.ln_a_ai[end_of_inflation_idx]
		self.inflation_idx = self.epsilon <= 1

		print(f'\nFound N_tot to be {self.N_tot:.3f} for the', self.name + 'model')

		if guess is not None:

			N_tot_relerr = np.abs(guess - self.N_tot) / np.abs(self.N_tot)
			print(f'with rel. err. of N_tot: {N_tot_relerr:.3e}\n')

		if plot:

			fig, ax = plt.subplots()

			ax.plot(self.tau[self.inflation_idx], self.epsilon[self.inflation_idx], label=r'$\epsilon$')

			if self.eta is not None:

				ax.plot(self.tau[self.inflation_idx], self.eta[self.inflation_idx], label=r'$\eta$')

			ax.text(text_xpos, text_ypos, r'$N_\mathrm{tot}=' + f'{self.N_tot:.3f}$')
			ax.legend()
			ax.set_xlabel(r'$\tau$')

			fig.savefig('figures/' + self.name + 'Ntot-compare-with-SRA.pdf', bbox_inches='tight')
			fig.savefig('figures/' + self.name + 'Ntot-compare-with-SRA.png', bbox_inches='tight')

	def pressure_energy_density_ratio(self, xlim=None):

		v = self.potential(self.psi, self.psi_i)

		p_phi = 0.5 * self.dpsi**2 - v
		rho_phi_c2 = 0.5 * self.dpsi**2 + v 
		w_phi = p_phi / rho_phi_c2

		fig, ax = plt.subplots()

		ax.plot(self.tau, w_phi)
		ax.set_xlabel(r'$\tau$')
		ax.set_ylabel(r'$\frac{p_\phi}{\rho_{\phi} c^2}$')
		ax.set_xlim(xlim)

		fig.savefig('figures/' + self.name + 'w_phi.pdf', bbox_inches='tight')
		fig.savefig('figures/' + self.name + 'w_phi.png', bbox_inches='tight')

	def epsilon_remainding_e_folds(self):

		idx = self.epsilon <= 1

		N_left = self.N_tot - self.ln_a_ai[idx]

		fig, ax = plt.subplots()

		ax.plot(N_left, self.epsilon[idx])
		ax.set_xscale('log')
		ax.set_xlabel(r'Remainding $e$-folds of inflation')
		ax.set_ylabel(r'$\epsilon$')
		ax.invert_xaxis()

		fig.savefig('figures/' + self.name + 'epsilon_remainding_efolds.pdf', bbox_inches='tight')
		fig.savefig('figures/' + self.name + 'epsilon_remainding_efolds.png', bbox_inches='tight')

	def plot_tensor_to_scalar_ratio(self):

		N = self.N_tot - self.ln_a_ai
		N_idx = np.logical_and(N >= 50, N <= 60)

		if self.eta is not None:

			n = 1 - 6 * self.epsilon[N_idx] + 2 * self.eta[N_idx]

		else:

			n = 1 - 4 * self.epsilon[N_idx] 

		r = 16 * self.epsilon[N_idx]

		x_text = np.max(n) - (np.max(n) - np.min(n)) / 2

		y_max = np.max([np.max(r), 0.044])

		fig, ax = plt.subplots()

		ax.plot(n, r)
		ax.fill_between(n, 0.044, alpha=0.25)
		ax.text(x_text, 0.02, r'$r<0.044$', ha='center')
		ax.set_xlabel('n')
		ax.set_ylabel('r')
		ax.set_xlim([np.min(n), np.max(n)])
		ax.set_ylim([0, 1.1 * y_max])

		fig.savefig('figures/' + self.name + 'n-r-plane.pdf', bbox_inches='tight')
		fig.savefig('figures/' + self.name + 'n-r-plane.png', bbox_inches='tight')
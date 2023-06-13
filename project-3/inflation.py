import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.integrate import simpson
from cycler import cycler

# Customizing the plots
plt.rcParams.update({
	'text.usetex': True,
	'font.family': 'Helvetica',
	'font.size': 12,
	'figure.figsize': (10.6, 5.3),
	'xtick.direction': 'inout',
	'ytick.direction': 'inout',
	'lines.linewidth': 1,
	'axes.prop_cycle': cycler(color=['k', 'r', 'royalblue'])
	})

class InflationModel:
	'''
	This class represents an inflation model. Needs potential, spatial derivative
	of potential, epsilon and eta, and initial value to create a specific model.
	'''
	def __init__(self, scalar_field_init_value, model_name):
		'''
		Constructor of the class. Initiates the object with initial
		value for the scalar field, and the name of the model.

		:param scalar_field_init_value:		Initial scalar field value, float
		:param model_name:					Name of the model, str
		'''
		self.psi_i = scalar_field_init_value
		self.name = str(model_name) + '_'

		self.tau_end = None		# Analytic tau_end is computed after epsilon is set
		self.eta = None			# Can be set manually, if not stays None

	def set_potential(self, potential):
		'''
		Set the potential for the model.

		:param potential: Potential for the model, callable
		'''
		self.potential = potential

	def set_potential_differential(self, pot_diff):
		'''
		Set the differential for the model.

		:param pot_diff: Derivative of the potential w.r.t. the field, callable
		'''
		self.pot_diff = pot_diff

	def double_time_derivative_scalar_field(self, psi, dpsi, h):
		'''
		The EoM for the scalar field, solved for d^2(psi)/dtau^2.

		:param psi:		Scalar field, float or array
		:param dpsi:	Derivative of scalar field, float or array
		:param h:		Hubble parameter, float or array
		'''
		dv = self.pot_diff(psi, self.psi_i)

		return - 3 * h * dpsi - dv

	def dimensionless_Hubble_parameter(self, psi, dpsi):
		'''
		Calculate the dimensionless Hubble parameter.

		:param psi:		Scalar field, array
		:param dpsi:	Time-derivative of the scalar field, array
		:return:		Dimensionless Hubble parameter, array
		'''
		v = self.potential(psi, self.psi_i)

		return np.sqrt(8 * np.pi / 3 * (1/2 * dpsi**2 + v))

	def solve_scalar_field(self, total_time, dtau=1e-2):
		'''
		Solve the equation of motion for the scalar field.
		If the model is the phi2 model, the SRA for psi,
		h^2, and ln(a/a_i) is calculated here.

		:param total_time:	End-time of integration, float
		:param dtau:		Time step, float
		'''
		N = int(np.ceil(total_time / dtau))
		tau = np.linspace(0, total_time, N+1)

		h = np.zeros(N+1)
		psi = np.zeros(N+1)
		dpsi = np.zeros(N+1)
		ln_a_ai = np.zeros(N+1)

		h[0] = 1; psi[0] = self.psi_i

		print('\nSolving the ' + self.name + 'model\n')

		for i in range(N):

			d2psi = self.double_time_derivative_scalar_field(psi[i], dpsi[i], h[i])

			ln_a_ai[i+1] = ln_a_ai[i] + h[i] * dtau
			dpsi[i+1] = dpsi[i] + d2psi * dtau
			psi[i+1] = psi[i] + dpsi[i+1] * dtau
			h[i+1] = self.dimensionless_Hubble_parameter(psi[i+1], dpsi[i+1])

		self.h = h; self.psi = psi; self.h2 = h**2
		self.dpsi = dpsi; self.ln_a_ai = ln_a_ai; self.tau = tau

		# The phi2-model has SRA solutions we can compare with
		if self.name == 'phi2_':

			self.psi_SRA = self.psi_i - 1 / (4 * np.pi * self.psi_i) * tau
			self.h2_SRA = (1 - 1 / (4 * np.pi * self.psi_i**2) * tau)**2
			self.ln_a_ai_SRA = tau - 1 / (8 * np.pi * self.psi_i**2) * tau**2

	def plot_solutions(self, compare_to_SRA=False, tight_layout=False, xlim=None):
		'''
		Plot the solutions from solve_scalar_field.

		:param compare_to_SRA:		boolean, True/False
		:param tight_layout:		boolean, True/False
		:param xlim:				Min. x-value and max. x-value, array
		'''
		filename = 'num-sols' if not compare_to_SRA else 'SRA-compare'

		tau, tau_end, psi = self.tau, self.tau_end, self.psi
		ln_a_ai, h2 = self.ln_a_ai, self.h2

		if xlim is not None:
			'''
			Zoom in on a given interval of tau.
			'''
			min_idx = np.where(tau >= xlim[0])[0][0]
			max_idx = np.where(tau <= xlim[1])[0][-1]

			tau = tau[min_idx:max_idx]; h2 = h2[min_idx:max_idx]
			ln_a_ai = ln_a_ai[min_idx:max_idx]; psi = psi[min_idx:max_idx]

		with plt.rc_context({'figure.figsize': (10.6, 6)}):

			fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

			ax1.plot(tau, psi/self.psi_i, label='Exact')
			ax1.set_ylabel(r'$\psi/\psi_i$')

			if xlim is not None or self.name == 'starobinsky_':

				axins = ax1.inset_axes([0.25, 0.25, 0.3, 0.5])
				axins.plot(tau, psi/self.psi_i)

				x1, x2, y1, y2 = 2697, 2713, -0.07, 0.07

				axins.set_xlim([x1, x2])
				axins.set_ylim([y1, y2])
				axins.tick_params(labelbottom=False)

				ax1.indicate_inset_zoom(axins, ec='black')

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
				ax1.set_ylim([- 0.1, 1.1])

				ax2.plot([tau_end, tau_end], [- 5, np.max(self.ln_a_ai) * 1.1], ls='dotted', color='black')
				ax2.set_ylim([- 5, np.max(self.ln_a_ai) * 1.1])

				ax3.plot([tau_end, tau_end], [- 0.1, 1.1], ls='dotted', color='black')
				ax3.set_ylim([- 0.1, 1.1])

			if self.name == 'phi2_' and not compare_to_SRA:

				axins = ax1.inset_axes([0.75, 0.2, 0.2, 0.7])
				axins.plot(tau, psi / self.psi_i)

				x1, x2, y1, y2 = 1010, 1410, -0.02, 0.02

				axins.set_xlim([x1, x2])
				axins.set_ylim([y1, y2])
				axins.tick_params(labelbottom=False)

				ax1.indicate_inset_zoom(axins, ec='black')

			if tight_layout:

				fig.tight_layout()

			else:

				fig.subplots_adjust(hspace=0.05)

			ax3.set_xlim(xlim)

			if xlim is not None:

				filename += '_zoomed'

			fig.savefig('figures/' + self.name + filename + '.pdf', bbox_inches='tight')
			fig.savefig('figures/' + self.name + filename + '.png', bbox_inches='tight')

	def set_epsilon_parameter(self, epsilon, print_tau_end=False):
		'''
		Set the function for calculating the SRA
		parameter epsilon. This will also set the
		end of inflation from when epsilon <= 1
		for the last time.

		:param epsilon:			Epsilon function, callable
		:param print_tau_end:	boolean, True/False
		'''
		self.epsilon = epsilon(self.psi)
		end_idx = np.where(self.epsilon <= 1)[0][-1]
		self.tau_end = self.tau[end_idx]

		if print_tau_end:

			print(f'\nTime when epsilon <= 1 for the last time: {self.tau_end}')

	def set_eta_parameter(self, eta):
		'''
		Set the function for the SRA parameter eta.

		:param eta: Eta function, callable
		'''
		self.eta = eta(self.psi)

	def num_e_folds(self, text_xpos=0, text_ypos=0.8, guess=500, plot=True):
		'''
		Calculate the total number of e-folds of
		inflation and plot the result (optional).

		:param text_xpos:	Where along the x-axis to put the text, float
		:param text_ypos:	Where along the y-axis to put the text, float
		:param guess:		Guess for N_tot, float/None
		:param plot:		boolean, Ture/False
		'''
		end_of_inflation_idx = np.where(self.epsilon >= 1)[0][0]

		self.N_tot = self.ln_a_ai[end_of_inflation_idx]
		self.inflation_idx = self.epsilon <= 1

		print(f'\nFound N_tot to be {self.N_tot:.3f} for the', self.name + 'model')

		if guess is not None:

			print(f'where we guessed N_tot to be {guess}')

		if plot:

			fig, ax = plt.subplots()

			ax.plot(self.tau[self.inflation_idx], self.epsilon[self.inflation_idx], label=r'$\epsilon$')

			if self.eta is not None:

				ax.plot(self.tau[self.inflation_idx], self.eta[self.inflation_idx], label=r'$\eta$')
				ax.legend()

			ax.text(text_xpos, text_ypos, r'$N_\mathrm{tot}=' + f'{self.N_tot:.3f}$')
			ax.set_xlabel(r'$\tau$')

			ylabel = r'$\epsilon$' if self.eta is None else None
			ax.set_ylabel(ylabel)

			fig.savefig('figures/' + self.name + 'Ntot-compare-with-SRA.pdf', bbox_inches='tight')
			fig.savefig('figures/' + self.name + 'Ntot-compare-with-SRA.png', bbox_inches='tight')

	def pressure_energy_density_ratio(self, zoomed=False):
		'''
		Calculate and plot the pressure to energy density ratio.

		:param zoomed:	boolean, True/False
		'''
		v = self.potential(self.psi, self.psi_i)

		p_phi = 0.5 * self.dpsi**2 - v
		rho_phi_c2 = 0.5 * self.dpsi**2 + v
		w_phi = p_phi / rho_phi_c2

		fig, ax = plt.subplots()

		ax.plot(self.tau, w_phi)
		ax.set_xlabel(r'$\tau$')
		ax.set_ylabel(r'$\frac{p_\phi}{\rho_{\phi} c^2}$')

		if zoomed:

			axins = ax.inset_axes([0.2, 0.3, 0.5, 0.4])
			axins.plot(self.tau, w_phi)

			x1, x2, y1, y2 = 2690, 2730, -1.05, 1.05

			axins.set_xlim([x1, x2])
			axins.set_ylim([y1, y2])
			axins.tick_params(labelbottom=False)

			ax.indicate_inset_zoom(axins, ec='black')

		fig.savefig('figures/' + self.name + 'w_phi.pdf', bbox_inches='tight')
		fig.savefig('figures/' + self.name + 'w_phi.png', bbox_inches='tight')

	def SRA_params_remaining_e_folds(self, include_approx=False):
		'''
		Plot the SRA parameters epsilon and eta as functions
		of e-folds. If epsilon=eta, i.e. self.eta=None, only
		the epsilon parameter will be plotted.

		:param include_approx: boolean, True/False
		'''
		save_name = self.name
		idx = self.epsilon <= 1

		N_left = self.N_tot - self.ln_a_ai[idx]

		if self.name == 'starobinsky_':

			y = - np.sqrt(16 * np.pi / 3) * self.psi

			N_SRA = 3/4 * np.exp(-y)
			epsilon_SRA = 3 / (4 * N_SRA**2)

			idx_SRA = epsilon_SRA <= 1

			N_tot_SRA = N_SRA[idx_SRA][0]

			eta_SRA = - 1/N_SRA

			print(f'\nN_tot from SRA: {N_tot_SRA:.3f}')

		fig, ax = plt.subplots()

		ax.plot(N_left, self.epsilon[idx], label=r'$\epsilon_\mathrm{exact}$')

		ax.set_xscale('log')
		ax.set_xlabel(r'Remaining $e$-folds of inflation')
		ylabel = r'$\epsilon$' if self.name == 'phi2_' else None
		ax.set_ylabel(ylabel)

		if self.eta is not None:

			ax.plot(N_left, self.eta[idx], label=r'$\eta_\mathrm{exact}$')

			if include_approx:

				ax.plot(N_SRA[idx_SRA], epsilon_SRA[idx_SRA], ls='dashed', color='black', label=r'$\epsilon\approx\frac{3}{4N^2}$')
				ax.plot(N_SRA[idx_SRA], eta_SRA[idx_SRA], ls='dashed', color='red', label=r'$\eta\approx-\frac{1}{N}$')

				save_name += 'with_approx_'

			ax.legend()

		ax.invert_xaxis()

		fig.savefig('figures/' + save_name + 'epsilon_remainding_efolds.pdf', bbox_inches='tight')
		fig.savefig('figures/' + save_name + 'epsilon_remainding_efolds.png', bbox_inches='tight')

	def plot_tensor_to_scalar_ratio(self, include_approx=False):
		'''
		Plot the tensor-to-scalar ration r as function of
		the scalar spectral index n. Also, the upper limit
		for r of 0.044 is included as a shaded region in the plot.
		For the Starobinsky model, the approximations can also
		be included in the plots to compare to the analytical results.

		:param include_approx: boolean, True/False
		'''
		save_name = self.name
		N = self.N_tot - self.ln_a_ai
		N_idx = np.logical_and(N >= 50, N <= 60)

		if self.eta is not None:

			n = 1 - 6 * self.epsilon[N_idx] + 2 * self.eta[N_idx]

		else:

			n = 1 - 4 * self.epsilon[N_idx]

		r = 16 * self.epsilon[N_idx]

		if not include_approx:

			print('\nr and n values for the ' + self.name + 'model:')
			print(f'r_min = {np.min(r):.3f}')
			print(f'r_max = {np.max(r):.3f}')
			print(f'n_min = {np.min(n):.4f}')
			print(f'n_max = {np.max(n):.4f}')

		r_max = 0.044
		n_min, n_max = 0.9607, 0.9691

		y_max = np.max([np.max(r), r_max])
		x_min = np.min([np.min(n), n_min])
		x_max = np.max([np.max(n), n_max])
		x = [x_min, x_max]

		x_text = x_max - (x_max - x_min) / 2
		xlabels = np.arange(float(f'{x_min:.3f}'), float(f'{x_max:.3f}'), 0.001)

		fig, ax = plt.subplots()

		ax.plot(n, r)

		if self.name == 'starobinsky_':

			y = - np.sqrt(16 * np.pi / 3) * self.psi

			N_SRA = 3/4 * np.exp(-y)
			N_idx_SRA = np.logical_and(N_SRA >= 50, N_SRA <= 60)

			n_SRA = 1 - 2 / N_SRA[N_idx_SRA]
			r_SRA = 12 / N_SRA[N_idx_SRA]**2

			if include_approx:

				print('\nr and n values for the ' + self.name + 'model approximation:')
				print(f'r_min = {np.min(r_SRA):.3f}')
				print(f'r_max = {np.max(r_SRA):.3f}')
				print(f'n_min = {np.min(n_SRA):.4f}')
				print(f'n_max = {np.max(n_SRA):.4f}\n')

				ax.plot(n_SRA, r_SRA, ls='dotted', label='Approx.')
				ax.legend()

				x_min = np.min([np.min(n), n_min, np.min(n_SRA)])

				save_name += 'with_approx_'

		ax.fill_between(x, r_max, alpha=0.15)
		ax.text(x_text, 0.025, r'$r<0.044$', ha='center')
		ax.text(x_text, 0.015, r'$0.9607<n<0.9691$', ha='center')
		ax.set_xlabel(r'$n$')
		ax.set_ylabel(r'$r$')
		ax.set_xlim([x_min - 0.0002, x_max + 0.0002])
		ax.set_xticks(xlabels)
		ax.set_ylim([0, 1.1 * y_max])

		fig.savefig('figures/' + save_name + 'n-r-plane.pdf', bbox_inches='tight')
		fig.savefig('figures/' + save_name + 'n-r-plane.png', bbox_inches='tight')

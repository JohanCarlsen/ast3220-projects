import numpy as np 
import matplotlib.pyplot as plt 
import scipy.constants as const
from scipy.integrate import simpson

plt.rcParams.update({
	'text.usetex': True,
	'font.family': 'Helvetica',
	'font.size': 12,
	'figure.figsize': (8, 4.5),
	'xtick.direction': 'inout',
	'ytick.direction': 'inout',
	'lines.linewidth': 1
	})

def potential(psi):

	v = 3 / (8 * np.pi) * (psi / psi_i)**2

	return v

def dvdpsi(psi):

	dv = 3 / (4 * np.pi) * psi / psi_i**2

	return dv 

def psi_double_dot(tau, psi, dpsi, h):

	dv = dvdpsi(psi)
	d2psi = - 3 *  h * dpsi - dv 

	return d2psi

def dimensionless_Hubble_parameter(psi, dpsi):

	v = potential(psi)

	res = np.sqrt(8 * np.pi / 3 * (1/2 * dpsi**2 + v))

	return res

def solve_scalar_field(dtau=1e-2):

	N = int(np.ceil(tau_end / dtau))
	tau = np.linspace(0, tau_end, N+1)

	h = np.zeros(N+1)
	psi = np.zeros(N+1)
	dpsi = np.zeros(N+1)
	ln_a_ai = np.zeros(N+1)

	h[0] = 1
	psi[0] = psi_i
	dpsi[0] = 0
	ln_a_ai[0] = 0

	for i in range(N):

		d2psi = psi_double_dot(tau[i], psi[i], dpsi[i], h[i])

		ln_a_ai[i+1] = ln_a_ai[i] + h[i] * dtau
		dpsi[i+1] = dpsi[i] + d2psi * dtau
		psi[i+1] = psi[i] + dpsi[i+1] * dtau
		h[i+1] = dimensionless_Hubble_parameter(psi[i+1], dpsi[i+1])

	h2 = h**2

	return tau, psi, dpsi, ln_a_ai, h2

# Constants
E_P = np.sqrt(const.hbar * const.c**5 / const.G) 	# Planck energy
m_P = np.sqrt(const.hbar * const.c / const.G) 		# Planck mass 
l_P = np.sqrt(const.hbar * const.G / const.c**3) 	# Planck length 
H_i = np.sqrt(8 * np.pi * const.G)

# Initial (i) and final (end) value of the scalar field phi
phi_i = 8.9251 * E_P
phi_end = E_P / (2 * np.sqrt(np.pi))

# Initial and final value of the dimensionless scalarfield psi
psi_i = phi_i / E_P
psi_end = phi_end / E_P

# End of inflation
tau_end = 4 * np.pi * psi_i**2 - 2 * np.sqrt(np.pi) / psi_i

tau, psi, dpsi, ln_a_ai, h2 = solve_scalar_field()

# SRA solutions
psi_SRA = psi_i - 1 / (4 * np.pi * psi_i) * tau
h2_SRA = (1 - 1 / (4 * np.pi * psi_i**2) * tau)**2
ln_a_ai_SRA = tau - 1 / (8 * np.pi * psi_i**2) * tau**2

epsilon = 1 / (4 * np.pi) * 1 / (psi**2)
SRA_idx = epsilon <= 1


with plt.rc_context({'figure.figsize': (10, 8)}):

	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

	ax1.plot(tau, psi_SRA, ls='dashed', color='red', label=r'$\psi_\mathrm{SRA}$')
	ax1.plot(tau, psi, ls='dotted', color='black', label=r'$\psi$')
	ax1.set_yscale('log')
	ax1.legend()

	ax2.plot(tau, ln_a_ai_SRA, ls='dashed', color='red', label=r'$\ln\left(\frac{a}{a_i}\right)_\mathrm{SRA}$')
	ax2.plot(tau, ln_a_ai, ls='dotted', color='black', label=r'$\ln\left(\frac{a}{a_i}\right)$')
	ax2.legend()

	ax3.plot(tau, h2_SRA, ls='dashed', color='red', label=r'$h^2_\mathrm{SRA}$')
	ax3.plot(tau, h2, ls='dotted', color='black', label=r'$h^2$')
	ax3.legend()
	ax3.set_yscale('log')
	ax3.set_xlabel(r'$\tau$')

	fig.subplots_adjust(hspace=0)
	fig.savefig('figures/SRA-compare.pdf')
	fig.savefig('figures/SRA-compare.png')

N_tot = - 4 * np.pi * simpson(psi[SRA_idx], psi[SRA_idx])
N_tot_relerr = np.abs(500 - N_tot) / np.abs(N_tot)

print(f'\nRel. err. of N_tot: {N_tot_relerr:.3e}')

fig, ax = plt.subplots()

ax.plot(tau[SRA_idx], epsilon[SRA_idx], color='black', label=r'$\epsilon$')
ax.text(0, 0.8, r'$N_{tot}=' + f'{N_tot:.3f}$')
ax.legend()
ax.set_xlabel(r'$\tau$')

fig.savefig('figures/Ntot-compare-with-SRA.pdf')
fig.savefig('figures/Ntot-compare-with-SRA.png')

plt.show()


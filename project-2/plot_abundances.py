import numpy as np
import matplotlib.pyplot as plt  
import astropy.constants as const
from astropy import units 
from scipy.integrate import solve_ivp
from functions import dYn_dYp, dYn_dYp_dYD, element_abundance

plt.rcParams.update({
	'text.usetex': True,
	'font.family': 'Helvetica',
	'font.size': 12,
	'figure.figsize': (10.6, 6),
	'xtick.direction': 'inout',
	'ytick.direction': 'inout'
	})

np.seterr(all='ignore')

# Constants
k = const.k_B.cgs.value											# Boltzmann's constant in cgs-units
hbar = const.hbar.cgs.value										# Planck's constant in cgs-units
c = const.c.cgs.value											# Speed of light in cgs-units
G = const.G.cgs.value											# Gravitational constant in cgs-units
m_e = const.m_e.cgs.value										# Electron mass in cgs-units 
m_p = const.m_p.cgs.value										# Proton mass in cgs-units
m_n = const.m_n.cgs.value										# Neutron mass in cgs-units
pi = np.pi
h = .7 															# Dimensionless Hubble constant
T0 = 2.725														# CMB temperature, today in K
H0 = (100 * h * units.km / (units.s * units.Mpc)).cgs.value 	# Hubble constant, today in s^-1
tau = 1700.														# Free neutron decay time [s]
q = 2.53														# (m_n - m_p) / m_e
rho_c0 = 3 * H0**2 / (8 * pi * G)

T_i = 100e9
T_f = 0.1e9 
T_range = [np.log(T_i), np.log(T_f)]

# Initial values 
Yn_i = 1 / (1 + np.exp((m_n - m_p) * c**2 / (k * T_i)))
Yp_i = 1 - Yn_i

initial_values = [Yn_i, Yp_i]

sol = solve_ivp(dYn_dYp, T_range, initial_values, method='Radau', rtol=1e-12, atol=1e-12, dense_output=True)

solT_init = sol.t[0]
solT_end = sol.t[-1]
T = np.linspace(solT_init, solT_end, 1001)

variables = sol.sol(T)
T_plot = np.exp(T)

Yn = variables[0, :]
Yp = variables[1, :]

# Equilibrium variables
Yn_equi = 1 / (1 + np.exp((m_n - m_p) * c**2 / (k * T_plot)))
Yp_equi = 1 - Yn_equi

fig, ax = plt.subplots()

ax.plot(T_plot, Yp, color='royalblue', label='p')
ax.plot(T_plot, Yp_equi, color='royalblue', ls='dotted')
ax.plot(T_plot, Yn, color='red', label='n')
ax.plot(T_plot, Yn_equi, color='red', ls='dotted')
ax.set_xlabel('T[K]')
ax.set_ylabel(r'$Y_i$')
ax.set_ylim([1e-3, 2])
ax.set_xlim([1e8, 1e11])
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
fig.savefig('figures/Yp_Yn.png')
fig.savefig('figures/Yp_Yn.pdf')

initial_values = [Yn_i, Yp_i, 0]

sol = solve_ivp(dYn_dYp_dYD, T_range, initial_values, method='Radau', rtol=1e-12, atol=1e-12, dense_output=True)

solT_init = sol.t[0]
solT_end = sol.t[-1]
T = np.linspace(solT_init, solT_end, 1001)

variables = sol.sol(T)
T_plot = np.exp(T)

Yn = variables[0, :]
Yp = variables[1, :]
YD = variables[2, :]

fig, ax = plt.subplots()

ax.plot(T_plot, Yp, color='royalblue', label='p')
ax.plot(T_plot, Yp_equi, color='royalblue', ls='dotted')
ax.plot(T_plot, Yn, color='red', label='n')
ax.plot(T_plot, Yn_equi, color='red', ls='dotted')
ax.plot(T_plot, 2 * YD, color='green', label='D')
ax.set_xlabel('T[K]')
ax.set_ylabel(r'Mass fraction $A_iY_i$')
ax.set_ylim([1e-3, 2])
ax.set_xlim([1e8, 1e11])
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
fig.savefig('figures/Yp_Yn_YD.png')
fig.savefig('figures/Yp_Yn_YD.pdf')

initial_values = [Yn_i, Yp_i, 0, 0, 0, 0, 0, 0]

T_i = 100e9
T_f = .01e9
T_range = [np.log(T_i), np.log(T_f)]

sol = solve_ivp(element_abundance, T_range, initial_values, method='Radau', rtol=1e-12, atol=1e-12, dense_output=True)

solT_init = sol.t[0]
solT_end = sol.t[-1]

T = np.linspace(solT_init, solT_end, 1001)
T_plot = np.exp(T)

variables = sol.sol(T)

Yn = variables[0, :]
Yp = variables[1, :]
YD = variables[2, :]
YT = variables[3, :]
YHe3 = variables[4, :]
YHe4 = variables[5, :]
YLi7 = variables[6, :]
YBe7 = variables[7, :]
sum_Yi = Yn + Yp + 2*YD + 3*YT + 3*YHe3 + 4*YHe4 + 7*YLi7 + 7*YBe7

fig, ax = plt.subplots()

ax.plot(T_plot, Yp, label='p')
ax.plot(T_plot, Yn, label='n')
ax.plot(T_plot, 2 * YD, label='D')
ax.plot(T_plot, 3 * YT, label='T')
ax.plot(T_plot, 3 * YHe3, label=r'He$^3$')
ax.plot(T_plot, 4 * YHe4, label=r'He$^4$')
ax.plot(T_plot, 7 * YLi7, label=r'Li$^7$')
ax.plot(T_plot, 7 * YBe7, label=r'Be$^7$')
ax.plot(T_plot, sum_Yi, lw=1, ls='dashed', color='black', label=r'$\sum A_iY_i$')
ax.set_xlabel('T[K]')
ax.set_ylabel(r'Mass fraction $A_iY_i$')
ax.set_ylim([1e-11, 10])
ax.set_xlim([1e7, 6e10])
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
fig.savefig('figures/element-abundances.png')
fig.savefig('figures/element-abundances.pdf')

plt.show()








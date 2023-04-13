import numpy as np
import matplotlib.pyplot as plt  
import astropy.constants as const
from scipy.integrate import solve_ivp, quad

def differentials(lnT, variables):
	'''
	Contains the differentials dY_i/d(lnT),
	which will be solved by solve_ivp.
	'''
	Yn, Yp = variables

	T = np.exp(lnT)
	T_nu = (4/11)**(1/3) * T 
	T9 = T * 1e-9
	T9_nu = T_nu * 1e-9
	H = H0 * np.sqrt(O_r0) * (T/T0)**2

	Z = 5.93 / T9 
	Z_nu = 5.93 / T9_nu

	a = 1. 
	b = np.inf

	func = lambda x, q: (x + q)**2 * (x**2 - 1)**(1/2) * x / ((1 + np.exp(x * Z)) * (1 + np.exp(-(x + q) * Z_nu))) \
	                  + (x - q)**2 * (x**2 - 1)**(1/2) * x / ((1 + np.exp(-x * Z)) * (1 + np.exp((x - q) * Z_nu)))

	I_n_p, _ = quad(func, a, b, args=(q,))
	I_p_n, _ = quad(func, a, b, args=(-q,))

	gamma_n_p = 1/tau * I_n_p
	gamma_p_n = 1/tau * I_p_n

	dYn = -1/H * (Yp * gamma_p_n - Yn * gamma_n_p)
	dYp = -1/H * (Yn * gamma_n_p - Yp * gamma_p_n)

	diffs = np.array([dYn, dYp])

	return diffs


# Constants 
k = const.k_B.cgs.value				# Boltzmann's constant in cgs-units
hbar = const.hbar.cgs.value			# Planck's constant in cgs-units
c = const.c.cgs.value				# Speed of light in cgs-units
G = const.G.cgs.value				# Gravitational constant in cgs-units
m_e = const.m_e.cgs.value			# Electron mass in cgs-units 
m_p = const.m_p.cgs.value			# Proton mass in cgs-units
m_n = const.m_n.cgs.value			# Neutron mass in cgs-units
Mpc = const.kpc.cgs.value * 1e3 	# Mpc in cgs-units
pi = np.pi 
h = .7 								# Dimensionless Hubble constant
Neff = 3. 
T0 = 2.725							# CMB temperature, today in K
H0 = 100e3 * h / Mpc				# Hubble constant, today in s^-1

# Radiation parameter today 
O_r0 = 8 * pi**3 * G / (45 * H0**2) * (k * T0)**4 / (hbar**3 * c**5) * (1 + Neff * 7/8 * (4/11)**(4/3))

def t(T):

	time = 1 / (2 * H0 * np.sqrt(O_r0)) * (T0/T)**2

	return time

print(f"{'Temperature [K]':<25}{'Time'}")

for i in range(10, 7, -1):

	temp = 10**i
	tot = t(temp)
	h = tot // 3600
	mins = (tot % 3600) // 60 
	sec = tot % 3600 % 60
	T = f"{temp:.0e}"

	print(f"{'':<5}{T:<14}{h:1.0f} h {mins:2.0f} min {sec:4.1f} s")


Ti = 100e9				# [K]
Tf = 1e8 				# [K]
T_range = [np.log(Ti), np.log(Tf)]
# T = np.linspace(Ti, Tf, 1001)

tau = 1700.		# Free neutron decay time [s]
q = 2.53		# (m_n - m_p) / m_e

Yn_i = 1 / (1 + np.exp((m_n - m_p) * c**2 / (k * Ti)))
Yp_i = 1 - Yn_i

initial_variables = [Yn_i, Yp_i]
sol = solve_ivp(differentials, T_range, initial_variables, method='Radau', rtol=1e-12, atol=1e-12, dense_output=True)

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

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(T_plot, Yp, color='red', label=r'$p$')
ax.plot(T_plot, Yp_equi, color='red', ls='dotted')
ax.plot(T_plot, Yn, color='royalblue', label=r'$n$')
ax.plot(T_plot, Yn_equi, color='royalblue', ls='dotted')
ax.set_xlabel('T [K]')
ax.set_ylabel(r'$Y_i$')
ax.legend()
ax.invert_xaxis()
ax.set_ylim([1e-3, 2])
ax.set_xscale('log')
ax.set_yscale('log')
fig.tight_layout()
plt.savefig('figures/fractions.pdf')
plt.savefig('figures/fractions.png')

plt.show()






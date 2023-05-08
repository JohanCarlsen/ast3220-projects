import numpy as np
import matplotlib.pyplot as plt  
import astropy.constants as const
from astropy import units 
from scipy.integrate import solve_ivp, quad
from time import perf_counter
from functions import element_abundance

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
rho_c0 = 3 * H0**2 / (8 * pi * G)								# Critical density today

# Initial and final temperatures 
T_i = 100e9
T_f = .01e9
T_range = [np.log(T_i), np.log(T_f)]

# Initial abundances of neutrons and protons 
Yn_i = 1 / (1 + np.exp((m_n - m_p) * c**2 / (k * T_i)))
Yp_i = 1 - Yn_i

initial_abundance = [Yn_i, Yp_i, 0, 0, 0, 0, 0, 0]		# All initial abundances other that p and n are zero 

n_runs = 20

Neff = np.linspace(1, 5, n_runs)
O_r0 = 8 * pi**3 * G / (45 * H0**2) * (k * T0)**4 / (hbar**3 * c**5) * (1 + Neff * 7/8 * (4/11)**(4/3))

# Using the best fit for Omega_b0
O_b0 = 0.0496

YD = np.zeros(n_runs)
Yp = np.zeros(n_runs)
YLi7 = np.zeros(n_runs)
YHe3 = np.zeros(n_runs)
YHe4 = np.zeros(n_runs)

print(f'\nComputing models for {n_runs} Neff values.')
print(f'Using O_b0 = {O_b0}\n')

start_outer = perf_counter()

for i in range(n_runs):

	sol = solve_ivp(element_abundance, T_range, initial_abundance, args=(O_b0, O_r0[i]), method='Radau', rtol=1e-12, atol=1e-12)

	_, Yp[i], YD[i], YT, YHe3_f, YHe4[i], YLi7_f, YBe7 = sol.y[:, -1]

	YHe3[i] = YHe3_f + YT 
	YLi7[i] = YLi7_f + YBe7

	end_inner = perf_counter()
	time_inner = end_inner - start_outer

	print(f'{((i+1) / n_runs) * 100:10.0f} % completed after: {time_inner//60:2.0f} min {time_inner%60:2.0f} s')

end_outer = perf_counter()

time_outer = end_outer - start_outer 

print(f'\nComputed models for {n_runs} Neff values in: {time_outer//60:2.0f} min {time_outer%60:.0f} s\n')

YDYp = YD / Yp 
YLi7Yp = YLi7 / Yp
YHe3 /= Yp
YHe4 *= 4

model = np.array([YDYp, YHe4, YLi7Yp])

np.save(f'fitting-Neff-model-{n_runs}-runs', model)
np.save(f'fitting-Neff-YHe3-{n_runs}-runs', YHe3)





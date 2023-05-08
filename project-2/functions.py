import numpy as np
import matplotlib.pyplot as plt  
import astropy.constants as const
from astropy import units 
from scipy.integrate import solve_ivp, simpson

def t(T):
	'''
	Calculate the age of the Universe at the temperature T.
	'''
	Neff = 3. 
	O_r0 = 8 * pi**3 * G / (45 * H0**2) * (k * T0)**4 / (hbar**3 * c**5) * (1 + Neff * 7/8 * (4/11)**(4/3))

	time = 1 / (2 * H0 * np.sqrt(O_r0)) * (T0/T)**2

	return time

def dYn_dYp(lnT, variables):
	'''
	Contains the differentials dY_i/d(lnT),
	which will be solved by solve_ivp.
	'''
	Yn, Yp = variables

	T = np.exp(lnT)
	T_nu = (4/11)**(1/3) * T 
	T9 = T * 1e-9
	T9_nu = T_nu * 1e-9
	Neff = 3. 
	O_r0 = 8 * pi**3 * G / (45 * H0**2) * (k * T0)**4 / (hbar**3 * c**5) * (1 + Neff * 7/8 * (4/11)**(4/3))
	H = H0 * np.sqrt(O_r0) * (T/T0)**2

	Z = 5.93 / T9 
	Z_nu = 5.93 / T9_nu

	func = lambda x, q: (x + q)**2 * (x**2 - 1)**(1/2) * x / ((1 + np.exp(x * Z)) * (1 + np.exp(-(x + q) * Z_nu))) \
	                  + (x - q)**2 * (x**2 - 1)**(1/2) * x / ((1 + np.exp(-x * Z)) * (1 + np.exp((x - q) * Z_nu)))

	x = np.linspace(1, 100, 1001)
	I1 = func(x, q)
	I2 = func(x, -q)

	I_n_p = simpson(I1, x)
	I_p_n = simpson(I2, x)

	# Rates for n <--> p 
	gamma_n_p = 1/tau * I_n_p
	gamma_p_n = 1/tau * I_p_n

	dYn = -1/H * (Yp * gamma_p_n - Yn * gamma_n_p)
	dYp = -1/H * (Yn * gamma_n_p - Yp * gamma_p_n)

	diffs = np.array([dYn, dYp])

	return diffs

def dYn_dYp_dYD(lnT, variables):
	'''
	Integrate function to calculate the
	fractional abundances of n, p, and D.
	'''
	Yn, Yp, YD = variables

	T = np.exp(lnT)
	T_nu = (4/11)**(1/3) * T 
	T9 = T * 1e-9
	T9_nu = T_nu * 1e-9
	Neff = 3. 
	O_r0 = 8 * pi**3 * G / (45 * H0**2) * (k * T0)**4 / (hbar**3 * c**5) * (1 + Neff * 7/8 * (4/11)**(4/3))
	H = H0 * np.sqrt(O_r0) * (T/T0)**2

	Z = 5.93 / T9 
	Z_nu = 5.93 / T9_nu

	O_b0 = .05
	rho_c0 = 3 * H0**2 / (8 * pi * G)
	rho_b = O_b0 * rho_c0 * (T/T0)**3

	func = lambda x, q: (x + q)**2 * (x**2 - 1)**(1/2) * x / ((1 + np.exp(x * Z)) * (1 + np.exp(-(x + q) * Z_nu))) \
	                  + (x - q)**2 * (x**2 - 1)**(1/2) * x / ((1 + np.exp(-x * Z)) * (1 + np.exp((x - q) * Z_nu)))

	x = np.linspace(1, 100, 1001)
	I1 = func(x, q)
	I2 = func(x, -q)

	I_n_p = simpson(I1, x)
	I_p_n = simpson(I2, x)

	pn =  2.5e4 * rho_b

	lmbda_n = 1./tau * I_n_p
	lmbda_p = 1./tau * I_p_n
	lmbda_D = 4.68e9 * pn / rho_b * T9**(3/2) * np.exp(-25.82 / T9)

	dYn = lmbda_n * Yn - lmbda_p * Yp - lmbda_D * YD + pn * Yn * Yp 
	dYp = lmbda_p * Yp - lmbda_n * Yn - lmbda_D * YD + pn * Yn * Yp 
	dYD = lmbda_D * YD - pn * Yn * Yp 

	diffs = np.array([dYn, dYp, dYD]) / H

	return diffs


# p + n <--> D + gamma
def strong_1(T_9, rho_b):


	rate_pn = 2.5e4 * rho_b
	rate_gammaD = 4.68e9 * rate_pn / rho_b * T_9**(3/2) * np.exp(-25.82 / T_9)

	return rate_pn, rate_gammaD

# p + D <--> He3 + gamma
def strong_2(T_9, rho_b):

	rate_pD = 2.23e3 * rho_b * T_9**(-2/3) * np.exp(-3.72 * T_9**(-1/3)) * (1 + 0.112 * T_9**(1/3) \
																			  + 3.38 * T_9**(2/3) \
																			  + 2.65 * T_9)
	rate_gammaHe3 = 1.63e10 * rate_pD / rho_b * T_9**(3/2) * np.exp(-63.75 / T_9)

	return rate_pD, rate_gammaHe3

# n + D <--> T + gamma
def strong_3(T_9, rho_b):

	rate_nD = rho_b * (75.5 + 1250 * T_9)
	rate_gammaT = 1.63e10 * rate_nD / rho_b * T_9**(3/2) * np.exp(-72.62 / T_9)

	return rate_nD, rate_gammaT

# n + He3 <--> p + T
def strong_4(T_9, rho_b):

	rate_nHe3_p = 7.06e8 * rho_b
	rate_pT_n = rate_nHe3_p * np.exp(-8.864 / T_9)

	return rate_nHe3_p, rate_pT_n

# p + T <--> He4 + gamma
def strong_5(T_9, rho_b):

	rate_pT_gamma = 2.87e4 * rho_b * T_9**(-2/3) * np.exp(-3.87 * T_9**(-1/3)) * (1 + 0.108 * T_9**(1/3) \
																					+ 0.466 * T_9**(2/3) \
																					+ 0.352 * T_9 \
																					+ 0.300 * T_9**(4/3) \
																					+ 0.576 * T_9**(5/3))
	rate_gamma_He4_p = 2.59e10 * rate_pT_gamma / rho_b * T_9**(3/2) * np.exp(-229.9 / T_9)

	return rate_pT_gamma, rate_gamma_He4_p

# n + He3 <--> He4 + gamma
def strong_6(T_9, rho_b):

	rate_nHe3_gamma = 6.0e3 * rho_b * T_9 
	rate_gamma_He4_n = 2.6e10 * rate_nHe3_gamma / rho_b * T_9**(3/2) * np.exp(-238.8 / T_9)

	return rate_nHe3_gamma, rate_gamma_He4_n

# D + D <--> n + He3
def strong_7(T_9, rho_b):

	rate_DD_n = 3.9e8 * rho_b * T_9**(-2/3) * np.exp(-4.26 * T_9**(-1/3)) * (1 + 0.0979 * T_9**(1/3) \
																			   + 0.642 * T_9**(2/3) \
																			   + 0.440 * T_9)
	rate_nHe3_D = 1.73 * rate_DD_n * np.exp(-37.94 / T_9)

	return rate_DD_n, rate_nHe3_D

# D + D <--> p + T
def strong_8(T_9, rho_b):

	rate_DD_p, _ = strong_7(T_9, rho_b)
	rate_pT_D = 1.73 * rate_DD_p * np.exp(-46.8 / T_9)

	return rate_DD_p, rate_pT_D

# D + D <--> He4 + gamma 
def strong_9(T_9, rho_b):

	rate_DD_gamma = 24.1 * rho_b * T_9**(-2/3) * np.exp(-4.26 * T_9**(-1/3)) * (T_9**(2/3) + 0.685 * T_9 \
																						   + 0.152 * T_9**(4/3) \
																						   + 0.265 * T_9**(5/3))
	rate_gammaHe4_D = 4.5e10 * rate_DD_gamma / rho_b * T_9**(3/2) * np.exp(-276.7 / T_9)

	return rate_DD_gamma, rate_gammaHe4_D

# D + He3 <--> He4 + p
def strong_10(T_9, rho_b):

	rate_DHe3 = 2.6e9 * rho_b * T_9**(-3/2) * np.exp(-2.99 / T_9)
	rate_He4p = 5.5 * rate_DHe3 * np.exp(-213.0 / T_9)

	return rate_DHe3, rate_He4p

# D + T <--> He4 + n 
def strong_11(T_9, rho_b):

	rate_DT = 1.38e9 * rho_b * T_9**(-3/2) * np.exp(-0.745 / T_9)
	rate_He4n = 5.5 * rate_DT * np.exp(-204.1 / T_9)

	return rate_DT, rate_He4n

# He3 + T <--> He4 + D
def strong_15(T_9, rho_b):

	rate_He3T_D = 3.88e9 * rho_b * T_9**(-2/3) * np.exp(-7.72 * T_9**(-1/3)) * (1 + 0.054 * T_9**(1/3))
	rate_He4D = 1.59 * rate_He3T_D * np.exp(-166.2 / T_9)

	return rate_He3T_D, rate_He4D

# He3 + He4 <--> Be7 + gamma 
def strong_16(T_9, rho_b):

	rate_He3He4 = 4.8e6 * rho_b * T_9**(-2/3) * np.exp(-12.8 * T_9**(-1/3)) * (1 + 0.0326 * T_9**(1/3) \
																				 - 0.219 * T_9**(2/3) \
																				 - 0.0499 * T_9 \
																				 + 0.0258 * T_9**(4/3) \
																				 + 0.0150 * T_9**(5/3))
	rate_gammaBe7 = 1.12e10 * rate_He3He4 / rho_b * T_9**(3/2) * np.exp(-18.42 / T_9)

	return rate_He3He4, rate_gammaBe7

# T + He4 <--> Li7 + gamma 
def strong_17(T_9, rho_b):

	rate_THe4 = 5.28e5 * rho_b * T_9**(-2/3) * np.exp(-8.08 * T_9**(-1/3)) * (1 + 0.0516 * T_9**(1/3))
	rate_gammaLi7 = 1.12e10 * rate_THe4 / rho_b * T_9**(3/2) * np.exp(-28.63 / T_9)

	return rate_THe4, rate_gammaLi7

# n + Be7 <--> p + Li7
def strong_18(T_9, rho_b):

	rate_nBe7_p = 6.74e9 * rho_b
	rate_pLi7_n = rate_nBe7_p * np.exp(-19.07 / T_9)

	return rate_nBe7_p, rate_pLi7_n

# p + Li7 <--> He4 + He4
def strong_20(T_9, rho_b):

	rate_pLi7_He4 = 1.42e9 * rho_b * T_9**(-2/3) * np.exp(-8.47 * T_9**(-1/3)) * (1 + 0.0493 * T_9**(1/3))
	rate_He4He4_p = 4.64 * rate_pLi7_He4 * np.exp(-201.3 / T_9)

	return rate_pLi7_He4, rate_He4He4_p

# n + Be7 <--> He4 + He4
def strong_21(T_9, rho_b):

	rate_nBe7_He4 = 1.2e7 * rho_b * T_9 
	rate_He4He4_n = 4.64 * rate_nBe7_He4 * np.exp(-220.4 / T_9)

	return rate_nBe7_He4, rate_He4He4_n


def element_abundance(lnT, variables, Omega_b0=None, Omega_r0=None):
	'''
	Integration function to calculate the element
	abundance for n, p, D, T, He3, He4, Li7, and Be7. 
	'''
	Yn, Yp, YD, YT, YHe3, YHe4, YLi7, YBe7 = variables

	T = np.exp(lnT)
	T_nu = (4/11)**(1/3) * T 
	T9 = T * 1e-9
	T9_nu = T_nu * 1e-9

	if Omega_r0 is None:

		Neff = 3. 
		O_r0 = 8 * pi**3 * G / (45 * H0**2) * (k * T0)**4 / (hbar**3 * c**5) * (1 + Neff * 7/8 * (4/11)**(4/3))

	else:

		O_r0 = Omega_r0

	H = H0 * np.sqrt(O_r0) * (T/T0)**2

	Z = 5.93 / T9 
	Z_nu = 5.93 / T9_nu

	if Omega_b0 is None:

		O_b0 = .05

	else:

		O_b0 = Omega_b0

	rho_b = O_b0 * rho_c0 * (T/T0)**3

	func = lambda x, q: (x + q)**2 * (x**2 - 1)**(1/2) * x / ((1 + np.exp(x * Z)) * (1 + np.exp(-(x + q) * Z_nu))) \
	                  + (x - q)**2 * (x**2 - 1)**(1/2) * x / ((1 + np.exp(-x * Z)) * (1 + np.exp((x - q) * Z_nu)))

	x = np.linspace(1, 100, 3501)
	I1 = func(x, q)
	I2 = func(x, -q)

	I_n_p = simpson(I1, x)
	I_p_n = simpson(I2, x)

	# Table 2a)
	# Reactions 1)-3)
	rate_wn = 1./tau * I_n_p
	rate_wp = 1./tau * I_p_n

	dYn = 0.0; dYHe3 = 0.0
	dYp = 0.0; dYHe4 = 0.0
	dYD = 0.0; dYLi7 = 0.0
	dYT = 0.0; dYBe7 = 0.0

	# Table 2a)
	# Reactions 1)-3)
	# n + nu_e <--> p + e 
	# n + (e+) <--> p + nu_e_bar 
	# 		 n <--> e + e + nu_e_bar 
	dYn 	-= 1/H * (- Yn * rate_wn + Yp * rate_wp)
	dYp 	-= 1/H * (  Yn * rate_wn - Yp * rate_wp) 

	# Table 2b)
	# Reaction 1)
	# p + n <--> D + gamma 
	rate_pn, rate_gammaD = strong_1(T9, rho_b)

	dYp		-= 1/H * (- rate_pn * Yn * Yp + rate_gammaD * YD) 
	dYn 	-= 1/H * (- rate_pn * Yn * Yp + rate_gammaD * YD) 
	dYD		-= 1/H * (  rate_pn * Yn * Yp - rate_gammaD * YD) 

	# Reaction 2)
	# p + D <--> He3 + gamma 
	rate_pD, rate_gammaHe3 = strong_2(T9, rho_b)

	dYp 	-= 1/H * (- rate_pD * Yp * YD + rate_gammaHe3 * YHe3) 
	dYD 	-= 1/H * (- rate_pD * Yp * YD + rate_gammaHe3 * YHe3) 
	dYHe3 	-= 1/H * (  rate_pD * Yp * YD - rate_gammaHe3 * YHe3) 

	# Reaction 3)
	# n + D <--> T + gamma 
	rate_nD, rate_gammaT = strong_3(T9, rho_b)

	dYn		-= 1/H * (- rate_nD * Yn * YD + rate_gammaT * YT) 
	dYD		-= 1/H * (- rate_nD * Yn * YD + rate_gammaT * YT) 
	dYT 	-= 1/H * (  rate_nD * Yn * YD - rate_gammaT * YT) 

	# Reaction 4)
	# n + He3 <--> p + T 
	rate_nHe3_p, rate_pT_n = strong_4(T9, rho_b)

	dYn		-= 1/H * (- rate_nHe3_p * Yn * YHe3 + rate_pT_n * Yp * YT) 
	dYHe3	-= 1/H * (- rate_nHe3_p * Yn * YHe3 + rate_pT_n * Yp * YT) 
	dYp 	-= 1/H * (  rate_nHe3_p * Yn * YHe3 - rate_pT_n * Yp * YT) 
	dYT 	-= 1/H * (  rate_nHe3_p * Yn * YHe3 - rate_pT_n * Yp * YT) 

	# Reaction 5) 
	# p + T <--> He4 + gamma 
	rate_pT_gamma, rate_gamma_He4_p = strong_5(T9, rho_b)

	dYp 	-= 1/H * (- rate_pT_gamma * Yp * YT + rate_gamma_He4_p * YHe4)
	dYT 	-= 1/H * (- rate_pT_gamma * Yp * YT + rate_gamma_He4_p * YHe4)
	dYHe4 	-= 1/H * (  rate_pT_gamma * Yp * YT - rate_gamma_He4_p * YHe4)

	# Reaction 6)
	# n + He3 <--> He4 + gamma 
	rate_nHe3_gamma, rate_gamma_He4_n = strong_6(T9, rho_b)

	dYn 	-= 1/H * (- rate_nHe3_gamma * Yn * YHe3 + rate_gamma_He4_n * YHe4) 
	dYHe3 	-= 1/H * (- rate_nHe3_gamma * Yn * YHe3 + rate_gamma_He4_n * YHe4) 
	dYHe4 	-= 1/H * (  rate_nHe3_gamma * Yn * YHe3 - rate_gamma_He4_n * YHe4) 

	# Reaction 7)
	# D + D <--> n + He3 
	rate_DD_n, rate_nHe3_D = strong_7(T9, rho_b)

	dYD 	-= 1/H * (   - rate_DD_n * YD * YD + 2 * rate_nHe3_D * Yn * YHe3)
	dYn 	-= 1/H * (.5 * rate_DD_n * YD * YD - 	 rate_nHe3_D * Yn * YHe3)
	dYHe3	-= 1/H * (.5 * rate_DD_n * YD * YD - 	 rate_nHe3_D * Yn * YHe3)

	# Reaction 8)
	# D + D <--> p + T 
	rate_DD_p, rate_pT_D = strong_8(T9, rho_b)

	dYD 	-= 1/H * (   - rate_DD_p * YD * YD + 2 * rate_pT_D * Yp * YT) 
	dYp 	-= 1/H * (.5 * rate_DD_p * YD * YD -	 rate_pT_D * Yp * YT) 
	dYT 	-= 1/H * (.5 * rate_DD_p * YD * YD -	 rate_pT_D * Yp * YT) 

	# Reaction 9)
	# D + D <--> He4 + gamma 
	rate_DD_gamma, rate_gammaHe4_D = strong_9(T9, rho_b)

	dYD 	-= 1/H * (   - rate_DD_gamma * YD * YD + 2 * rate_gammaHe4_D * YHe4)
	dYHe4 	-= 1/H * (.5 * rate_DD_gamma * YD * YD - 	 rate_gammaHe4_D * YHe4)

	# Reaction 10)
	# D + He3 <--> He4 + p 
	rate_DHe3, rate_He4p = strong_10(T9, rho_b)

	dYD 	-= 1/H * (- rate_DHe3 * YD * YHe3 + rate_He4p * YHe4 * Yp)
	dYHe3 	-= 1/H * (- rate_DHe3 * YD * YHe3 + rate_He4p * YHe4 * Yp)
	dYHe4	-= 1/H * (  rate_DHe3 * YD * YHe3 - rate_He4p * YHe4 * Yp)
	dYp 	-= 1/H * (  rate_DHe3 * YD * YHe3 - rate_He4p * YHe4 * Yp)

	# Reaction 11)
	# D + T <--> He4 + n
	rate_DT, rate_He4n = strong_11(T9, rho_b)

	dYD 	-= 1/H * (- rate_DT * YD * YT + rate_He4n * YHe4 * Yn) 
	dYT 	-= 1/H * (- rate_DT * YD * YT + rate_He4n * YHe4 * Yn) 
	dYHe4 	-= 1/H * (  rate_DT * YD * YT - rate_He4n * YHe4 * Yn) 
	dYn 	-= 1/H * (  rate_DT * YD * YT - rate_He4n * YHe4 * Yn) 

	# Reaction 15)
	# He3 + T <--> He4 + D 
	rate_He3T_D, rate_He4D = strong_15(T9, rho_b)

	dYHe3 	-= 1/H * (- rate_He3T_D * YHe3 * YT + rate_He4D * YHe4 * YD)
	dYT 	-= 1/H * (- rate_He3T_D * YHe3 * YT + rate_He4D * YHe4 * YD)
	dYHe4 	-= 1/H * (  rate_He3T_D * YHe3 * YT - rate_He4D * YHe4 * YD)
	dYD 	-= 1/H * (  rate_He3T_D * YHe3 * YT - rate_He4D * YHe4 * YD)

	# Reaction 16)
	# He3 + He4 <--> Be7 + gamma 
	rate_He3He4, rate_gammaBe7 = strong_16(T9, rho_b)

	dYHe3 	-= 1/H * (- rate_He3He4 * YHe3 * YHe4 + rate_gammaBe7 * YBe7)
	dYHe4 	-= 1/H * (- rate_He3He4 * YHe3 * YHe4 + rate_gammaBe7 * YBe7)
	dYBe7 	-= 1/H * (  rate_He3He4 * YHe3 * YHe4 - rate_gammaBe7 * YBe7)

	# Reaction 17)
	# T + He4 <--> Li7 + gamma 
	rate_THe4, rate_gammaLi7 = strong_17(T9, rho_b)

	dYT 	-= 1/H * (- rate_THe4 * YT * YHe4 + rate_gammaLi7 * YLi7)
	dYHe4 	-= 1/H * (- rate_THe4 * YT * YHe4 + rate_gammaLi7 * YLi7)
	dYLi7 	-= 1/H * (  rate_THe4 * YT * YHe4 - rate_gammaLi7 * YLi7)

	# Reaction 18)
	# n + Be7 <--> p + Li7
	rate_nBe7_p, rate_pLi7_n = strong_18(T9, rho_b)

	dYn 	-= 1/H * (- rate_nBe7_p * Yn * YBe7 + rate_pLi7_n * Yp * YLi7)
	dYBe7 	-= 1/H * (- rate_nBe7_p * Yn * YBe7 + rate_pLi7_n * Yp * YLi7)
	dYp 	-= 1/H * (  rate_nBe7_p * Yn * YBe7 - rate_pLi7_n * Yp * YLi7)
	dYLi7 	-= 1/H * (  rate_nBe7_p * Yn * YBe7 - rate_pLi7_n * Yp * YLi7)

	# Reaction 20)
	# p + Li7 <--> He4 + He4
	rate_pLi7_He4, rate_He4He4_p = strong_20(T9, rho_b)

	dYp 	-= 1/H * (  - rate_pLi7_He4 * Yp * YLi7 + .5 * rate_He4He4_p * YHe4 * YHe4)
	dYLi7 	-= 1/H * (  - rate_pLi7_He4 * Yp * YLi7 + .5 * rate_He4He4_p * YHe4 * YHe4)
	dYHe4 	-= 1/H * (2 * rate_pLi7_He4 * Yp * YLi7 - 	   rate_He4He4_p * YHe4 * YHe4)

	# Reaction 21)
	# n + Be7 <--> He4 + He4
	rate_nBe7_He4, rate_He4He4_n = strong_21(T9, rho_b)

	dYn 	-= 1/H * (  - rate_nBe7_He4 * Yn * YBe7 + .5 * rate_He4He4_n * YHe4 * YHe4)
	dYBe7 	-= 1/H * (  - rate_nBe7_He4 * Yn * YBe7 + .5 * rate_He4He4_n * YHe4 * YHe4)
	dYHe4 	-= 1/H * (2 * rate_nBe7_He4 * Yn * YBe7 -  	   rate_He4He4_n * YHe4 * YHe4)

	diffs = np.array([dYn, dYp, dYD, dYT, dYHe3, dYHe4, dYLi7, dYBe7])

	return diffs

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

if __name__ == '__main__':

	print(f"\n{'Temperature [K]':<25}{'Time'}")

	for i in range(10, 7, -1):

		temp = 10**i
		tot = t(temp)
		h = tot // 3600
		mins = (tot % 3600) // 60 
		sec = tot % 3600 % 60
		T = f"{temp:.0e}"

		print(f"{'':<5}{T:<14}{h:1.0f} h {mins:2.0f} min {sec:4.1f} s")

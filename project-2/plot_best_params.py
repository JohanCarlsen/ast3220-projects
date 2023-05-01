import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
from functions import gaussian

plt.rcParams.update({
	'text.usetex': True,
	'font.family': 'Helvetica',
	'font.size': 12,
	'figure.figsize': (7, 7),
	'xtick.direction': 'inout',
	'ytick.direction': 'inout'
	})

'''
Finding the best fit for Omega_b0
'''
model = np.load('fitting-O_b0-model-20-runs.npy')
YHe3 = np.load('fitting-O_b0-YHe3-20-runs.npy')

YDYp, YHe4, YLi7Yp = model

log_YDYp, log_YHe4, log_YLi7Yp = np.log(model)
log_YHe3 = np.log(YHe3)

n_runs = len(YDYp)

O_b0 = np.linspace(0.01, 1, n_runs)
log_O_b0 = np.log(O_b0)

inter_YDYp = interp1d(log_O_b0, log_YDYp, kind='cubic')
inter_YHe4 = interp1d(log_O_b0, log_YHe4, kind='cubic')
inter_YLi7Yp = interp1d(log_O_b0, log_YLi7Yp, kind='cubic')
inter_YHe3 = interp1d(log_O_b0, log_YHe3, kind='cubic')

O_b0 = np.linspace(0.01, 1, 1001)

YDYp = np.exp(inter_YDYp(np.log(O_b0)))
YHe4 = np.exp(inter_YHe4(np.log(O_b0)))
YLi7Yp = np.exp(inter_YLi7Yp(np.log(O_b0)))
YHe3 = np.exp(inter_YHe3(np.log(O_b0)))

model = np.array([YDYp, YHe4, YLi7Yp])

obs_YDYp = 2.57e-5
obs_YDYp_lower = (2.57 - .03) * 1e-5
obs_YDYp_upper = (2.57 + .03) * 1e-5

obs_4YHe4 = 0.254
obs_4YHe4_lower = 0.254 - .003
obs_4YHe4_upper = 0.254 + .003

obs_YLi7Yp = 1.6e-10 
obs_YLi7Yp_lower = (1.6 - .3) * 1e-10
obs_YLi7Yp_upper = (1.6 + .3) * 1e-10

obser = np.array([obs_YDYp, obs_4YHe4, obs_YLi7Yp])
error = np.array([.03e-5, .003, .3e-10])

chi_sq = np.sum((model.T - obser)**2 / error**2, axis=-1)

best_O_b0 = O_b0[np.where(chi_sq == np.min(chi_sq))[0][0]]

prob = gaussian(O_b0, best_O_b0, error)

gridspec = dict(height_ratios=[.3, 1, .3])
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw=gridspec, sharex=True)

ax1.plot(O_b0, 4 * YHe4, color='green', label=r'He$^4$')
ax1.fill_between(O_b0, obs_4YHe4_lower, obs_4YHe4_upper, color='green', alpha=.4)
ax1.plot([best_O_b0, best_O_b0], [.2, .33], ls='dotted', lw=1, color='black')
ax1.set_ylim([.2, .33])
ax1.set_yticks(np.linspace(.2, .3, 3))
ax1.set_ylabel(r'$4Y_{He^4}$')
ax1.set_xscale('log')
ax1.legend(loc='upper left')

ax2.plot(O_b0, YDYp, label='D')
ax2.plot(O_b0, YHe3, label=r'He$^3$')
ax2.plot(O_b0, YLi7Yp, color='red', label=r'Li$^7$')
ax2.fill_between(O_b0, obs_YLi7Yp_lower, obs_YLi7Yp_upper, color='red', alpha=.4)
ax2.fill_between(O_b0, obs_YDYp_lower, obs_YDYp_upper, color='royalblue', alpha=.4)
ax2.plot([best_O_b0, best_O_b0], [1.5e-11, 1e-3], ls='dotted', lw=1, color='black')
ax2.set_ylim([1.5e-11, 1e-3])
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylabel(r'$Y_i/Y_p$')
ax2.legend(loc='center left')

ax3.plot(O_b0, prob/np.max(prob), lw=1, color='black')
ax3.set_xlim([1e-2, 1])
ax3.set_yticks(np.linspace(0, 1, 3))
ax3.set_ylabel('Normalized\nprobability')
ax3.set_xscale('log')
ax3.set_xlabel(r'$\Omega_{b0}$')

fig.savefig('figures/best-fit-O_r0.pdf')
fig.savefig('figures/best-fit-O_r0.png')

'''
Finding the best fit for Neff. Note that in order to find the best fit for both 
Omega_b0 and Neff, we first computed the above fit for Omega_b0, and then used 
that value in the program fitting_Neffpy. 
'''
model = np.load('fitting-Neff-model-20-runs.npy')
YHe3 = np.load('fitting-Neff-YHe3-20-runs.npy')

YDYp, YHe4, YLi7Yp = model

log_YDYp, log_YHe4, log_YLi7Yp = np.log(model)
log_YHe3 = np.log(YHe3)

n_runs = len(YDYp)

Neff = np.linspace(1, 5, n_runs)
log_Neff = np.log(Neff)

inter_YDYp = interp1d(log_Neff, log_YDYp, kind='cubic')
inter_YHe4 = interp1d(log_Neff, log_YHe4, kind='cubic')
inter_YLi7Yp = interp1d(log_Neff, log_YLi7Yp, kind='cubic')
inter_YHe3 = interp1d(log_Neff, log_YHe3, kind='cubic')

Neff = np.linspace(1, 5, 1001)

YDYp = np.exp(inter_YDYp(np.log(Neff)))
YHe4 = np.exp(inter_YHe4(np.log(Neff)))
YLi7Yp = np.exp(inter_YLi7Yp(np.log(Neff)))
YHe3 = np.exp(inter_YHe3(np.log(Neff)))

model = np.array([YDYp, YHe4, YLi7Yp])

chi_sq = np.sum((model.T - obser)**2 / error**2, axis=-1)

best_Neff = Neff[np.where(chi_sq == np.min(chi_sq))[0][0]]

prob = gaussian(Neff, best_Neff, error)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

ax1.plot(Neff, 4 * YHe4, color='green', label=r'He$^4$')
ax1.fill_between(Neff, obs_4YHe4_lower, obs_4YHe4_upper, color='green', alpha=.4)
ax1.plot([best_Neff, best_Neff], [.22, .30], ls='dotted', lw=1, color='black')
ax1.set_ylim([.22, .30])
ax1.set_yticks(np.linspace(.22, .3, 5))
ax1.set_ylabel(r'$4Y_{He^4}$')
ax1.legend()

ax2.plot(Neff, YDYp, label='D')
ax2.plot(Neff, YHe3, label=r'He$^3$')
ax2.fill_between(Neff, obs_YDYp_lower, obs_YDYp_upper, alpha=.4)
ax2.plot([best_Neff, best_Neff], [.5e-5, 5e-5], ls='dotted', lw=1, color='black')
ax2.set_ylabel(r'$Y_i/Y_p$')
ax2.set_ylim([.5e-5, 5e-5])
ax2.set_yticks(np.linspace(1, 5, 5)*1e-5)
ax2.legend()

ax3.plot(Neff, YLi7Yp, color='red', label=r'Li$^7$')
ax3.fill_between(Neff, obs_YLi7Yp_lower, obs_YLi7Yp_upper, color='red', alpha=.4)
ax3.plot([best_Neff, best_Neff], [5e-11, 2e-10], ls='dotted', lw=1, color='black')
ax3.set_ylabel(r'$Y_i/Y_p$')
ax3.set_ylim([5e-11, 2e-10])
ax3.set_yticks(np.linspace(.5, 2, 4)*1e-10)
ax3.legend(loc='lower left')

ax4.plot(Neff, prob/np.max(prob), color='black', lw=1)
ax4.set_xlim([1, 5])
ax4.set_ylabel('Normalized\nprobability')
ax4.set_xlabel(r'N$_{eff}$')

fig.savefig('figures/best-fit-Neff.png')
fig.savefig('figures/best-fit-Neff.pdf')

plt.show()
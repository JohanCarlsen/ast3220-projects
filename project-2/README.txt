All of the functions are located in the functions.py program.
The other programs imports the necessary functions as needed. 

Run the functions.py program in order to compute the age of the 
Universe at T=10e10 K, T=10e9 K, and T=10e8 K.

All of the programs (except functions.py) saves figures to the 
figures/ folder. 

IMPORTANT NOTE:
The programs fitting_*.py stores the needed arrays as .npy arrays.
They are then imported into the plot_best_params.py program, and 
it is then neccessary to first compute the models for fitting 
Omega_b0, and then use this value when computing the best fit for 
N_eff. Uncomment the exit() in plot_best_params.py if the best 
value for Omega_b0 is not yet found.
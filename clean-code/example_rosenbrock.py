#example of using DAS to optimize modified rosenbrock function (defined in paper)

import numpy as np
import DASOptimizer



rosen_beta = 4.0

#fitness function
def f(x):
	x = x.reshape(D,-1)
	return np.exp(-rosen_beta*np.sum((1 - x[::2,:])**2 + 100*(x[1::2,:] - x[::2,:]**2)**2, axis = 0))

#noisy samples (Bernoulli distribution)
def sample(x):
	B = x.shape[1]
	
	samp = 1*(f(x) >= np.random.rand(B))
	return samp


D = 2

#initialize optimizer with rough estimate of fitness optimum
opt = DASOptimizer.DAS(sample, D, initial_est = 0.5)

#DAS hyperparameters
#normalized time step (update of position relative to sampling window width)
opt.dt0 = 0.5 
#batch size
opt.B = 100

#option of diagonal sampling window matrix (window is specified by O(d) instead of O(d^2) real numbers)
opt.diag = False


#inital window (circular centered at origin)
L_init = np.ones(D)
if(not opt.diag):
	L_init = np.diag(L_init)

opt.init_window(np.zeros(D), L_init)

#optimize
xw, info = opt.optimize(1000*100, verbose = True)






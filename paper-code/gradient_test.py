#this code checks that the gradient estimator matches the real gradient for a toy problem


import numpy as np
import matplotlib.pyplot as plt

import SGD_sampler_ellipsoid as samp

D = 2

#pyramidal cost fucntion
def f(x):
	#return np.ones(x.shape[1])
	return np.maximum(np.min(1 - np.abs(x), axis =0), 0)

#smoothed cost function
grid_N = 50
def h(xw, L):
	L_inv = np.linalg.inv(L)
	
	z_rng = np.arange(-2,2,2/grid_N)
	X,Y = np.meshgrid(z_rng,z_rng)
	
	
	z = np.zeros((2, len(z_rng), len(z_rng)))
	z[0,:,:] = X
	z[1,:,:] = Y
	z = z.reshape(2, len(z_rng)**2)
	x = xw.reshape(2,1) + np.dot(L, z)
	w = np.exp(-np.sum(z**2, axis = 0)/2)
	
	
	return np.sum(w*f(x))/np.sum(w)


xw_init = np.array([0.5,1.5])
L_init = np.diag(np.array([1,1.5]))


h_list = []
f_list = []
x_range = np.arange(-3,3,0.05)
for x in x_range:
	h_list.append(h(np.array([x,0]), L_init))
	F = f(np.array([x,0]))
	print(F, 1 - np.abs(x))
	f_list.append(F)

plt.plot(x_range, h_list)
plt.plot(x_range, f_list)
plt.show()

plt.close()


xw_real = []
L_real = []
dt = 0.01

delta = 0.001

xw = xw_init*1
L = L_init*1

for i in range(2000):
	
	dL = np.zeros((D,D))
	dx = np.zeros(D)
	
	for i1 in range(D):
		for i2 in range(D):
			#compute gradient with respect to L using finite difference
			diff = np.zeros((D,D))
			diff[i1,i2]  = 1
			
			dL[i1,i2] += (h(xw, L + delta*diff) - h(xw, L - delta*diff))/(2*delta)
	
	for i1 in range(D):
		diff = np.zeros(D)
		diff[i1] = 1
		dx[i1] = (h(xw + delta*diff, L) - h(xw -  delta*diff, L))/(2*delta)
	
	L = L + dt*dL
	xw = xw + dt*dx
	
	xw_real.append(xw)
	L_real.append(L)



#no noise for now
sigma = 0.0

def sample(x):
	return f(x) + np.random.randn(x.shape[1])*sigma

#use stochastic gradient descent sampler	
SGD_samp = samp.Sampler(sample, D)

SGD_samp.init_window(xw_init, L_init)
SGD_samp.dt = 0.001

tot_samp_rec, xw_rec, L_rec = SGD_samp.optimize(tot_samp_max = 50000)

f_list = [f(x) for x in xw_rec]
window_size = [np.sum(np.diag(L)) for L in L_rec]


x_pos = [xw[0] for xw in xw_rec]
x2_pos = [xw[1] for xw in xw_rec]
y_pos = [L[1,1] for L in L_rec]

plt.plot(x_pos, x2_pos)
plt.plot([xw[0] for xw in xw_real], [xw[1] for xw in xw_real])

plt.show()
plt.close()

plt.plot(x_pos, y_pos)
plt.plot([xw[0] for xw in xw_real], [L[1,1] for L in L_real])

plt.show()
plt.close()

plt.plot(tot_samp_rec, f_list)
plt.show()
plt.close()

plt.xscale("log")
plt.yscale("log")
plt.plot(tot_samp_rec, 1-np.array(f_list))
plt.show()
plt.close()

plt.plot(tot_samp_rec, window_size)
plt.show()
plt.close()



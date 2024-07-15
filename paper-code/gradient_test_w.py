#this code checks that the gradient estimator matches the real gradient for a toy problem
#calculate exact trajectory with finite differences + calculate noisy trajectory of sampler

import numpy as np
import matplotlib.pyplot as plt

import SGD_sampler as samp
import toy_funcs

import sys
import os



#script parameters
D = 2

#no noise for now
sigma = 0.0


xw_init = np.array([0.2, 1.0])
w_init = 0.5



func_idx = 3


#outpath
outdir = sys.argv[0].split(".")[0]

outdir = outdir + "/D_%i_sigma_%f_w_init_%f_func_%i" % (D, sigma, w_init, func_idx)

if(not os.path.exists("figs/" + outdir)):
	os.makedirs("figs/" + outdir)

#save/show plot
plot_idx = 0
def show_plot():
	global plot_idx
	plt.savefig("figs/" + outdir + "/fig" + str(plot_idx) + ".png")
	plot_idx += 1
	
	if(False):
		plt.show()


#load fitness fucntion
if(func_idx == 0):
	f = toy_funcs.f0
if(func_idx == 1):
	f = toy_funcs.f1
if(func_idx == 2):
	f = toy_funcs.f2
if(func_idx == 3):
	f = toy_funcs.f3



#smoothed cost function (uses sum over grid mesh to approximate integral)
grid_N = 50
def h(xw, w):
		
	z_rng = np.arange(-4,4*(1  + 1/grid_N),2/grid_N)
	X,Y = np.meshgrid(z_rng,z_rng)
	
	
	z = np.zeros((2, len(z_rng), len(z_rng)))
	z[0,:,:] = X
	z[1,:,:] = Y
	z = z.reshape(2, len(z_rng)**2)
	x = xw.reshape(2,1) + w*z
	weights = np.exp(-np.sum(z**2, axis = 0)/2)
	
	return np.sum(weights*f(x))/np.sum(weights)





#display fitness function
z_rng = np.arange(-2,2,0.02)
X_,Y_ = np.meshgrid(z_rng,z_rng)

z = np.zeros((2, len(z_rng), len(z_rng)))
z[0,:,:] = X_
z[1,:,:] = Y_
F_ = f(z)

plt.title("fitness function")
plt.pcolormesh(X_,Y_, F_)

circlex = np.cos(np.arange(0,2,0.01)*np.pi)
circley = np.sin(np.arange(0,2,0.01)*np.pi)

plt.plot(xw_init[0] + circlex*w_init, xw_init[1] + circley*w_init, color = "red", label = "initial sampling window")

plt.colorbar()
plt.legend()
show_plot()
plt.close()


h_list = []
f_list = []
x_range = np.arange(-3,3,0.05)
for x in x_range:
	h_list.append(h(np.array([x,0]), w_init))
	F = f(np.array([x,0]))
	#print(F, 1 - np.abs(x))
	f_list.append(F)


plt.title("fitness function slice")
plt.xlabel("$x_0")
plt.ylabel("$f(x_0, 0)$ or $h(w, x_0, 0)$")
plt.plot(x_range, h_list, label = "smoothed fitness function w = " + str(w_init))
plt.plot(x_range, f_list, label = "fitness function")

plt.legend()

show_plot()

plt.close()


xw_real = []
w_real = []
dt = 0.01

delta = 0.001

xw = xw_init*1
w = w_init*1

#calculate "exact" trajectory using finite differences
print("calculating exact traj...")
for i in range(1000):
	
	
	dx = np.zeros(D)
	
	
	dw = (h(xw, w + delta) - h(xw, w - delta))/(2*delta)
	
	for i1 in range(D):
		diff = np.zeros(D)
		diff[i1] = 1
		dx[i1] = (h(xw + delta*diff, w) - h(xw -  delta*diff, w))/(2*delta)
	
	scale = w**2
	w = w + dt*dw*scale/(2*D)
	xw = xw + dt*dx*scale
	
	xw_real.append(xw)
	w_real.append(w)
	
	#print("finite difference gradient", dx, dw)




def sample(x):
	return f(x) + np.random.randn(x.shape[1])*sigma

#use stochastic gradient descent sampler	
SGD_samp = samp.Sampler(sample, D)

SGD_samp.init_window(xw_init, w_init)
SGD_samp.dt = 0.01

print("calculating sampled traj...")
tot_samp_rec, xw_rec, w_rec = SGD_samp.optimize(tot_samp_max = 50000)

f_list = [f(x) for x in xw_rec]
window_size = w_rec

x_pos = [xw[0] for xw in xw_rec]
x2_pos = [xw[1] for xw in xw_rec]

plt.xlabel("$x_0$")
plt.ylabel("$x_1$")

plt.pcolormesh(X_,Y_, F_)

plt.plot(x_pos, x2_pos, label = "sampler trajectory")
plt.plot([xw[0] for xw in xw_real], [xw[1] for xw in xw_real],  label = "true trajectory")

plt.colorbar()
plt.legend()
show_plot()
plt.close()

plt.xlabel("$x_0$")
plt.ylabel("$w$")
plt.plot(x_pos, w_rec, label = "sampler trajectory")
plt.plot([xw[0] for xw in xw_real], w_real, label = "true trajectory")

plt.legend()

show_plot()
plt.close()

plt.xlabel("number of samples")
plt.ylabel("fitness")
plt.plot(tot_samp_rec, f_list)
show_plot()
plt.close()


plt.xlabel("number of samples")
plt.ylabel("$1-f(x_w)$")

plt.xscale("log")
plt.yscale("log")
plt.plot(tot_samp_rec, 1-np.array(f_list))
show_plot()
plt.close()

plt.xlabel("number of samples")
plt.ylabel("$w$")
plt.plot(tot_samp_rec, w_rec)
show_plot()
plt.close()



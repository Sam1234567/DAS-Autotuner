#this code checks the asymptotic scaling of the error with respect to number of samples

import numpy as np
import matplotlib.pyplot as plt

import SGD_sampler_ellipsoid as samp
import toy_funcs

import sys
import os



#script parameters
D = 40
if(len(sys.argv) > 1):
	D = int(sys.argv[1])

#no noise for now
sigma = 0.1
noise_type = 0
#0 = gaussian noise
#1 = Bernoulli noise


xw_init = np.array([2.0,2.0] + [2.0]*(D-2))
xw_init = np.array([0,0.5]*int(D/2))
xw_init = np.random.rand(D)
#xw_init = np.array([0.0, 0.25]*int(D/2))
xw_init = np.array([2.33]*D)
#xw_init = np.array([0.5,0.5]*int(D/2))
w_init = 2.0
g = 0.0
g_exp = 0.0
kappa = 0.5

dt = 0.1
nsamp_max = 40*1000*1000

func_idx = 3
f_opt = 1

fixed_w = False

init_mode = -1

#2 12 0.1 1 0 0.25 0.5 0.0 0.0 0.5 40000000

#2 3 0.1 0 1 2.0 0.5 0.0 0.0 0.1 40000000

if(len(sys.argv) > 2):
	func_idx = int(sys.argv[2])
	
if(len(sys.argv) > 3):
	sigma = float(sys.argv[3])

if(len(sys.argv) > 4):
	noise_type = int(sys.argv[4])

if(len(sys.argv) > 5):
	init_mode = int(sys.argv[5])
	
if(len(sys.argv) > 6):
	w_init = float(sys.argv[6])
	
if(len(sys.argv) > 7):
	kappa = float(sys.argv[7])
	
if(len(sys.argv) > 8):
	g = float(sys.argv[8])
	
if(len(sys.argv) > 9):
	g_exp = float(sys.argv[9])
	
if(len(sys.argv) > 10):
	dt = float(sys.argv[10])
	
if(len(sys.argv) > 11):
	nsamp_max = int(sys.argv[11])

#different initial window positions used in different figures
if(init_mode == 0):
	xw_init = np.random.rand(D)
if(init_mode == 1):
	xw_init = np.array([2.33]*D)

#outpath
outdir = sys.argv[0].split(".")[0]


if(not fixed_w):
	outdir = outdir + "/D_%i_sigma_%f_w_init_%f_func_%i" % (D, sigma, w_init, func_idx)
else:
	outdir = outdir + "/D_%i_sigma_%f_w_init_%f_func_%i_fw" % (D, sigma, w_init, func_idx)

if(g > 0):
	outdir = outdir + "_g_%f_%f" % (g, g_exp)

if(kappa != 1.0):
	outdir = outdir + "_kappa_%f" % (kappa)


if(noise_type > 0):
	outdir = outdir + "_noise_%i" % noise_type


if(not os.path.exists("figs/" + outdir)):
	os.makedirs("figs/" + outdir)
	
if(not os.path.exists("data/" + outdir)):
	os.makedirs("data/" + outdir)

#save/show plot
plot_idx = 0
def show_plot():
	global plot_idx
	plt.savefig("figs/" + outdir + "/fig" + str(plot_idx) + ".png")
	plot_idx += 1
	
	if(False):
		plt.show()

#save data
def save_data(name, data):
	dat_idx = 0
	while(os.path.exists("data/" + outdir + "/" + name + str(dat_idx) + ".txt")):
		dat_idx += 1
	
	np.savetxt("data/" + outdir + "/" + name + str(dat_idx) + ".txt", data)


#load fitness fucntion

f = toy_funcs.get_func(func_idx)








#display fitness function
z_rng = np.arange(-3,3,0.02)
X_,Y_ = np.meshgrid(z_rng,z_rng)

z = np.zeros((D, len(z_rng), len(z_rng)))
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

plt.title("fitness function (log)")
plt.pcolormesh(X_,Y_, np.log(np.abs(F_)))

circlex = np.cos(np.arange(0,2,0.01)*np.pi)
circley = np.sin(np.arange(0,2,0.01)*np.pi)

plt.plot(xw_init[0] + circlex*w_init, xw_init[1] + circley*w_init, color = "red", label = "initial sampling window")

plt.colorbar()
plt.legend()
show_plot()
plt.close()








xw = xw_init*1
w = w_init*1

def sample(x):
	if(noise_type == 0):
		noise = np.random.randn(x.shape[1])*sigma
		#print(noise)
		out = f(x) + noise
		#print(x, "sample", out)
	if(noise_type == 1):
		noise = np.random.rand(x.shape[1])
		#print(noise)
		out = (noise < f(x))*1.0
		
	return out

#use stochastic gradient descent sampler	
SGD_samp = samp.Sampler(sample, D)
#growth of window
SGD_samp.g = g
SGD_samp.g_exp = g_exp
SGD_samp.kappa = kappa

SGD_samp.init_window(xw_init, np.diag([w_init]*D))
if(fixed_w):
	SGD_samp.dt = dt
else:
	SGD_samp.dt = dt
SGD_samp.fixed_w = fixed_w

print("calculating sampled traj...")
tot_samp_rec, xw_rec, L_rec = SGD_samp.optimize(tot_samp_max = nsamp_max)

w_rec = [np.average(L**2)**0.5 for L in L_rec]
L_ravel = [np.ravel(L) for L in L_rec]

f_list = [f(x) for x in xw_rec]

save_data("samps_f_w", np.array([tot_samp_rec, f_list, w_rec])[:,::10].T)
save_data("xw", np.array(xw_rec)[::10,:])
save_data("L", np.array(L_ravel)[::50,:])


window_size = w_rec


#plots

x_pos = [xw[0] for xw in xw_rec]
x2_pos = [xw[1] for xw in xw_rec]

plt.xlabel("$x_0$")
plt.ylabel("$x_1$")

plt.pcolormesh(X_,Y_, F_)

plt.plot(x_pos, x2_pos, label = "sampler trajectory")

plt.colorbar()
plt.legend()
show_plot()
plt.close()



plt.xlabel("number of samples")
plt.ylabel("fitness")
plt.plot(tot_samp_rec, f_list)
show_plot()
plt.close()


plt.xlabel("number of samples")
plt.ylabel("$" + str(f_opt) + "-f(x_w)$")

plt.xscale("log")
plt.yscale("log")
for c in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
	plt.plot(tot_samp_rec, np.array(tot_samp_rec)**(-0.5)*c, color = "gray", dashes = [3,3])

plt.plot(tot_samp_rec, f_opt-np.array(f_list))
show_plot()
plt.close()

plt.xlabel("number of samples")
plt.ylabel("$w$")
plt.plot(tot_samp_rec, w_rec)
show_plot()
plt.close()


plt.xlabel("number of samples")
plt.ylabel("$w$")
plt.xscale("log")
plt.yscale("log")

plt.plot(tot_samp_rec, w_rec)
show_plot()
plt.close()



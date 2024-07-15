#this code checks the asymptotic scaling of the error with respect to number of samples

import numpy as np
import matplotlib.pyplot as plt

import SGD_sampler as samp
import toy_funcs

import sys
import os



#script parameters
D = 5

if(len(sys.argv) > 1):
	D = int(sys.argv[1])

#no noise for now
sigma = 0.1


#xw_init = np.array([2.0,2.0] + [2.0]*(D-2))
xw_init = np.array([-1.2,1]*int(D/2))
xw_init = np.array([0,0]*int(D/2))
xw_init = np.array([1,1]*int(D/2))
xw_init = np.array([1.0,2.33]*int(D/2))
#xw_init = np.array([2.33]*D)
w_init = 1.0



func_idx = 11
f_opt = 1

fixed_w = False

noise_type = 1


#outpath
outdir = sys.argv[0].split(".")[0]


if(not fixed_w):
	outdir = outdir + "/D_%i_sigma_%f_w_init_%f_func_%i" % (D, sigma, w_init, func_idx)
else:
	outdir = outdir + "/D_%i_sigma_%f_w_init_%f_func_%i_fw" % (D, sigma, w_init, func_idx)

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








xw = xw_init*1
w = w_init*1

def sample(x):
	noise = np.random.randn(x.shape[1])*sigma
	#print(noise)
	out = f(x) + noise
	#print(x, "sample", out)
	return out

#use stochastic gradient descent sampler	
SGD_samp = samp.Sampler(sample, D)

SGD_samp.init_window(xw_init, w_init)
if(fixed_w):
	SGD_samp.dt = 0.0001
else:
	SGD_samp.dt = 0.1
SGD_samp.fixed_w = fixed_w

print("calculating sampled traj...")
tot_samp_rec, xw_rec, w_rec = SGD_samp.optimize(tot_samp_max = 5000000)



f_list = [f(x) for x in xw_rec]

save_data("samps_f_w", np.array([tot_samp_rec, f_list, w_rec])[:,::10].T)
save_data("xw", np.array(xw_rec)[::10,:])


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



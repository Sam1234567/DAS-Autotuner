#this code checks the asymptotic scaling of the error with respect to number of samples

import numpy as np
import matplotlib.pyplot as plt

import SGD_sampler as samp
import toy_funcs

#import noisyopt 

import noisyopt


import sys
import os



#script parameters
D = 2

if(len(sys.argv) > 1):
	D = int(sys.argv[1])

#no noise for now
sigma = 0.1


#xw_init = np.array([2.0,2.0] + [2.0]*(D-2))
xw_init = np.array([2.33]*D)
xw_init = np.array([0.0,0.5]*int(D/2))
xw_init = np.random.rand(D)

#xw_init = np.array([0.0, 0.5]*int(D/2))
w_init = 2.0



func_idx = 12
noise_type = 1

method = "SPSA"
#method = "DS"


init_mode = -1

#2 12 0.1 1 0 0.25 40000000

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
	nsamp_max = int(sys.argv[7])

#different initial window positions used in different figures
if(init_mode == 0):
	xw_init = np.random.rand(D)
if(init_mode == 1):
	xw_init = np.array([2.33]*D)

#outpath
outdir = sys.argv[0].split(".")[0]



outdir = outdir + "/D_%i_sigma_%f_w_init_%f_func_%i_%s" % (D, sigma, w_init, func_idx, method)

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

tot_samp_rec = []
xw_rec = []
w_rec = []

eval_count = 0

def obj(x):
	global eval_count
	eval_count += 1
	R_eval = 1
	if(eval_count % 100 == 0):
		tot_samp_rec.append(eval_count*R_eval)
		xw_rec.append(x)
		w_rec.append(1)
	if(eval_count % 100 == 0):
		print(eval_count, f(x), x)
	x_extended = np.outer(x, np.ones(R_eval))
	return -np.average(sample(x_extended))

if(method == "SPSA"):
	res, x_list = noisyopt.minimizeSPSA(obj, x0=xw_init, niter = 500000, c=w_init, a = 0.1, alpha = 0.6, gamma= 0.3, paired=False)
if(method == "DS"):
	res, x_list = noisyopt.minimizeCompass(obj, x0=xw_init, niter = 250000, errorcontrol = True, deltatol=0.05, feps = 0.01, paired=False)

print(x_list)
xw_rec = x_list
tot_samp_rec = (1 + np.array(range(len(x_list))))*2*100
w_rec = [1]*len(xw_rec)

f_list = [f(x) for x in xw_rec]

save_data("samps_f_w", np.array([tot_samp_rec, f_list, w_rec])[:,::10].T)
save_data("xw", np.array(xw_rec)[::10,:])


window_size = w_rec


#plots

x_pos = [xw[0] for xw in xw_rec]
x2_pos = [xw[1] for xw in xw_rec]

plt.xlabel("$x_0$")
plt.ylabel("$x_1$")

#plt.pcolormesh(X_,Y_, F_)

plt.plot(x_pos, x2_pos, label = "sampler trajectory")

#plt.colorbar()
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
for c in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
	plt.plot(tot_samp_rec, np.array(tot_samp_rec)**(-0.5)*c, color = "gray", dashes = [3,3])

plt.plot(tot_samp_rec, 1-np.array(f_list))
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



#this code checks the asymptotic scaling of the error with respect to dimension
#uses output of convergence_test.py

import numpy as np
import matplotlib.pyplot as plt

import SGD_sampler as samp
import toy_funcs

import sys
import os

fontsize = 12

#script parameters
D_list = [2,5,10,20,40]
D_list = [5, 10,20]
D_list = [5, 10, 20]
D_list = [2, 5, 10, 20, 40]

D_list = [5, 10, 20, 40]

D_list = [2, 5, 10, 20, 40]
D_list = [2, 4, 8, 20]


#nnoise
sigma = 0.1


# xw_init = np.array([0.2, 2.5] + [0.5]*(D-2))
w_init = 2.0
g = 0
kappa = 0.5


func_idx = 3

if(len(sys.argv) > 1):
	kappa = float(sys.argv[1])


#outpath
outdir = sys.argv[0].split(".")[0]

outdir = outdir + "/sigma_%f_w_init_%f_func_%i" % (sigma, w_init, func_idx)

if(kappa != 1.0):
	outdir = outdir + "_kappa_%f" % (kappa)



if(not os.path.exists("figs/" + outdir)):
	os.makedirs("figs/" + outdir)
	
if(not os.path.exists("data/" + outdir)):
	os.makedirs("data/" + outdir)
	

indir = "convergence_test_ellipsoid"
	
#indir = "convergence_test"
indir = indir + "/D_%s_sigma_%f_w_init_%f_func_%i" % ("%i", sigma, w_init, func_idx)

if(g > 0):
	indir = indir + "_g_%f_%f" % (g, g_exp)

if(kappa != 1.0):
	indir = indir + "_kappa_%f" % (kappa)


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










plt.xlabel("number of samples")
plt.ylabel("$1-f(x_w)$")

plt.xscale("log")
plt.yscale("log")

samp_range = np.array([10,50000000])
for c in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
	plt.plot(samp_range, np.array(samp_range)**(-0.5)*c, color = "gray", dashes = [3,3])

samp_max = 0

color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

interp_range = np.exp(np.arange(np.log(samp_range[0]), np.log(samp_range[1]), 0.1))

averages = []

for D, color in zip(D_list, color_list):
	indir_ = indir % D
	run_idx = 0
	
	
	sum = 0
	
	print("data/" + indir_ + "/samps_f_w%i.txt" % run_idx)
	while(os.path.exists("data/" + indir_ + "/samps_f_w%i.txt" % run_idx)):
		(tot_samp_rec, f_list, w_rec) = np.loadtxt("data/" + indir_ + "/samps_f_w%i.txt" % run_idx).T
		plt.plot(tot_samp_rec, 1-np.array(f_list), label = "D = " + str(D) if run_idx == 0 else None, color = color, alpha = 0.5)
		sum = sum + np.interp(interp_range, tot_samp_rec, 1 - np.array(f_list))
		run_idx += 1
	print(run_idx)
	if(run_idx == 0):
		run_idx = 1
	averages.append(sum/run_idx)

plt.legend(fontsize = fontsize)
show_plot()
plt.close()



plt.xlabel("number of samples")
plt.ylabel("$1-f(x_w)$")

plt.xscale("log")
plt.yscale("log")

samp_range = np.array([10,50000000])
for c in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
	plt.plot(samp_range, np.array(samp_range)**(-0.5)*c, color = "gray", dashes = [3,3])


for D, avg, color in zip(D_list, averages, color_list):
	
	plt.plot(interp_range, avg , label = "D = " + str(D))
	

plt.legend(fontsize = fontsize)
show_plot()
plt.close()


plt.xlabel("number of samples")
plt.ylabel("$(1-f(x_w))/D$")

plt.xscale("log")
plt.yscale("log")

for c in [1.4]:
	plt.plot(samp_range, np.array(samp_range)**(-0.5)*c, color = "gray", dashes = [3,3])


for D, avg, color in zip(D_list, averages, color_list):
	
	plt.plot(interp_range, avg/D , label = "D = " + str(D))
	

plt.legend(fontsize = fontsize)
show_plot()
plt.close()


plt.xlabel("(number of samples)/D")
plt.ylabel("$(1-f(x_w))}$")

plt.xscale("log")
plt.yscale("log")

for c in [0.5, 1.0,2.0, 4.0, 8.0, 16.0]:
	plt.plot(samp_range, np.array(samp_range)**(-0.5)*c, color = "gray", dashes = [3,3])


for D, avg, color in zip(D_list, averages, color_list):
	
	plt.plot(interp_range/D, avg , label = "D = " + str(D))
	

plt.legend(fontsize = fontsize)
show_plot()
plt.close()





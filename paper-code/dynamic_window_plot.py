#this code tests the tuner on a specific SAT instance 

import numpy as np
import matplotlib.pyplot as plt

import SGD_sampler as samp
import toy_funcs

import SAT_CAC as satcac
import InstanceUtils as inst_uils
import random

import time
import sys
import os



#script parameters
D = 5

if(len(sys.argv) > 1):
	D = int(sys.argv[1])

#no noise for now

#xw_init = np.array([2.0,2.0] + [2.0]*(D-2))
xw_init = np.array([0.7]*D)
w_init_list = [2.0, 1.0, 0.5, 0.25]
w_init_dynamic = 2.0


func_idx = 3
noise_type = 0


fixed_w = False

sigma = 1.0

#outpath
outdir = sys.argv[0].split(".")[0]





outdir = outdir + "/D_%i_fidx_%i" % (D, func_idx)



indir = "convergence_test"

if(not fixed_w):
	indir = indir + "/D_%i_sigma_%f_w_init_%f_func_%i" % (D, sigma, w_init_dynamic, func_idx)
else:
	indir = indir + "/D_%i_sigma_%f_w_init_%f_func_%i_fw" % (D, sigma, w_init_dynamic, func_idx)

if(noise_type > 0):
	indir = indir + "_noise_%i" % noise_type





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
	
	if(True):
		plt.show()

#save data
def save_data(name, data):
	dat_idx = 0
	while(os.path.exists("data/" + outdir + "/" + name + str(dat_idx) + ".txt")):
		dat_idx += 1
	
	np.savetxt("data/" + outdir + "/" + name + str(dat_idx) + ".txt", data)


dat_idx = 0

xw_list = []
t_list = []
f_list = []
w_list = []

print(indir)
while(os.path.exists("data/" + indir + "/xw%i.txt" % dat_idx)):
	
	xw_list.append(np.loadtxt("data/" + indir + "/xw%i.txt" % dat_idx))
	t_list.append(np.loadtxt("data/" + indir + "/samps_f_w%i.txt" % dat_idx)[:,0])
	f_list.append(np.loadtxt("data/" + indir + "/samps_f_w%i.txt" % dat_idx)[:,1])
	w_list.append(np.loadtxt("data/" + indir + "/samps_f_w%i.txt" % dat_idx)[:,2])
	
	dat_idx += 1

print(t_list)

xw_list_fixed = []
t_list_fixed = []
f_list_fixed = []
w_list_fixed = []

for w_init in w_init_list:
	indir = "convergence_test_gasnikov"

	if(not fixed_w):
		indir = indir + "/D_%i_sigma_%f_w_init_%f_func_%i" % (D, sigma, w_init, func_idx)
	else:
		indir = indir + "/D_%i_sigma_%f_w_init_%f_func_%i_fw" % (D, sigma, w_init, func_idx)

	if(noise_type > 0):
		indir = indir + "_noise_%i" % noise_type
	xw_list_fixed.append(np.loadtxt("data/" + indir + "/xw%i.txt" % 0))
	t_list_fixed.append(np.loadtxt("data/" + indir + "/samps_f_w%i.txt" % 0)[:,0])
	f_list_fixed.append(np.loadtxt("data/" + indir + "/samps_f_w%i.txt" % 0)[:,1])
	w_list_fixed.append(np.loadtxt("data/" + indir + "/samps_f_w%i.txt" % 0)[:,2])

t_range = 10**np.arange(1,6,0.1)
average_f = np.zeros(len(t_range))
min_f = 100*np.ones(len(t_range))
max_f = -100*np.ones(len(t_range))

for i in range(0, len(t_list)):
	f_interp =  np.interp(t_range, t_list[i], f_list[i])
	average_f += f_interp
	min_f = np.minimum(f_interp, min_f)
	max_f = np.maximum(f_interp, max_f)

average_f = average_f/len(t_list)
	
if(len(t_list) > 0):
	#plt.plot(t_list[0], 1 - f_list[0], color = "blue", alpha = 0.2, label = "DIS")
	#for i in range(1, len(t_list)):
		#plt.plot(t_list[i], 1 - f_list[i], alpha = 0.2,color = "blue")

	
	plt.plot(t_range, 1 - average_f, label = "DIS (this work)")
	plt.fill_between(t_range, 1 - min_f, 1 - max_f, alpha = 0.2 , color = "blue")

for w, t, f in zip(w_init_list, t_list_fixed, f_list_fixed):
	plt.plot(t,1-f, label = "Gasnikov 2022, w = " + str(w))


plt.yscale("log")
plt.xscale("log")
plt.xlabel("number of samples")
plt.ylabel(r"$f_{opt} - f(x)$")
plt.legend()
show_plot()
plt.close()


if(len(t_list) > 0):
	plt.plot(t_list[0], 1 - f_list[0], color = "blue", alpha = 0.2, label = "DIS")
	for i in range(1, len(t_list)):
		plt.plot(t_list[i], 1 - f_list[i], alpha = 0.2,color = "blue")

	plt.plot(t_range, 1 - average_f, label = "DIS (average)")

for w, t, f in zip(w_init_list, t_list_fixed, f_list_fixed):
	plt.plot(t,1-f, label = "Gasnikov 2022, w = " + str(w))


plt.yscale("log")
plt.xscale("log")
plt.xlabel("number of samples")
plt.ylabel(r"$f_{opt} - f(x)$")
plt.legend()
show_plot()
plt.close()








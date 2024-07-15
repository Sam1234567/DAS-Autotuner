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

fontsize = 12

#script parameters
D = 4

if(len(sys.argv) > 1):
	D = int(sys.argv[1])

#no noise for now

#xw_init = np.array([2.0,2.0] + [2.0]*(D-2))
xw_init = np.array([0.7]*D)
w_init = 0.5

w_init_noisyopt = 1.0

func_idx = 5

ellips = False
gasnikov = False
noisyopt = True
method = "SPSA"

fixed_w = False

N = 150
M = int(4.0*N)
K = 3

dt = 0.5
nsamp_max = 50000

T = np.exp(5.0)


#outpath
outdir = sys.argv[0].split(".")[0]




if(not fixed_w):
	outdir = outdir + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i" % (D, w_init, dt, nsamp_max)
else:
	outdir = outdir + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i_fw" % (D, w_init, dt, nsamp_max)




	
outdir = outdir + "/N_%i_M_%i_K_%i_T_%f" % (N,M,K,T)


indir = ""
if(not fixed_w):
	indir = indir + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i" % (D, w_init, dt, nsamp_max)
else:
	indir = indir + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i_fw" % (D, w_init, dt, nsamp_max)

indir = indir + "/N_%i_M_%i_K_%i_T_%f" % (N,M,K,T)

indir_noisyopt = ""
if(not fixed_w):
	indir_noisyopt = indir_noisyopt + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i" % (D, w_init_noisyopt, dt, nsamp_max)
else:
	indir_noisyopt = indir_noisyopt + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i_fw" % (D, w_init_noisyopt, dt, nsamp_max)

indir_noisyopt = indir_noisyopt + "_" + str(method)

indir_noisyopt = indir_noisyopt + "/N_%i_M_%i_K_%i_T_%f" % (N,M,K,T)

indir_list = []

indir_list.append("SAT_test_ensemble_ellipsoid" + indir)
indir_list.append("SAT_test_ensemble" + indir)
indir_list.append("SAT_test_ensemble_gasnikov" + indir)

indir_list.append("SAT_test_ensemble_noisyopt" + indir_noisyopt)
indir_list.append("SAT_test_ensemble_bohb" + indir)


label_list = [ "DAS (this work)", "DIS (this work)", "Gasnikov 2022", "SPSA (Spall 1998)", "BOHB (Falkner 2018)"]


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

xw_list_all = []
t_list_all = []
f_list_all = []
w_list_all = []

for indir in indir_list:
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
	
	xw_list_all.append(xw_list)
	t_list_all.append(t_list)
	f_list_all.append(f_list)
	w_list_all.append(w_list)
	

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



for idx in range(len(indir_list)):
	
	for t, f in zip(t_list_all[idx], f_list_all[idx]):
		plt.plot(t, f, color = colors[idx])
	
	plt.plot([], [], color = colors[idx], label = label_list[idx])

plt.xlabel("number of samples")
plt.ylabel("sucess rate")
plt.xlim(0, 50000)
plt.legend(loc = "upper left", fontsize =  fontsize)

show_plot()
plt.close()


for idx in range(len(indir_list)):
	t_range = np.arange(0, 50000, 10)
	f_sum = np.zeros(len(t_range))
	for t, f in zip(t_list_all[idx], f_list_all[idx]):
		f_sum += np.interp(t_range, t, f)
	
	
	plt.plot(t_range, f_sum/len(t_list_all[idx]), color = colors[idx], label = label_list[idx])

plt.xlabel("number of samples")
plt.ylabel("(average) sucess rate")
plt.xlim(0, 50000)
plt.legend(loc = "upper left", fontsize =  fontsize)

show_plot()
plt.close()

for idx in range(len(indir_list)):
	t_range = np.arange(0, 50000, 10)
	f_sum = np.zeros(len(t_range))
	f_var = np.zeros(len(t_range))
	f_max = -100*np.ones(len(t_range))
	f_min = 100*np.ones(len(t_range))
	for t, f in zip(t_list_all[idx], f_list_all[idx]):
		f_interp = np.interp(t_range, t, f)
		f_sum += f_interp
		f_var += f_interp**2
		f_max = np.maximum(f_interp, f_max)
		f_min = np.minimum(f_interp, f_min)
		print(f_min, f_interp)
		
	
	f_avg = f_sum/len(t_list_all[idx])
	f_var = f_var/len(t_list_all[idx]) - f_avg**2
	if(len(t_list_all[idx]) > 0):
		plt.fill_between(t_range, f_avg - np.sqrt(f_var), f_avg + np.sqrt(f_var), color = colors[idx], alpha = 0.2)
		plt.plot(t_range,f_avg , color = colors[idx], label = label_list[idx])
	
plt.xlabel("number of samples")
plt.ylabel("(average) fitness")
plt.xlim(0, 50000)
plt.legend(loc = "upper left", fontsize =  fontsize)

show_plot()
plt.close()

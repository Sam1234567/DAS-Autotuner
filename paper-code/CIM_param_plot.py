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
D = 4

if(len(sys.argv) > 1):
	D = int(sys.argv[1])

#no noise for now

#xw_init = np.array([2.0,2.0] + [2.0]*(D-2))
xw_init = np.array([0.7]*D)
w_init = 2.0



func_idx = 5

ellips = True
gasnikov = False
noisyopt = False
method = "SPSA"

fixed_w = False

N = 150


dt = 0.5
nsamp_max = 50000

temp_beta = 0.01

T = np.exp(5.0)


#outpath
outdir = sys.argv[0].split(".")[0]

if(len(sys.argv) > 2):
	N = int(sys.argv[2])

if(len(sys.argv) > 3):
	temp_beta = float(sys.argv[3])

	
if(len(sys.argv) > 4):
	dt = float(sys.argv[4])

if(len(sys.argv) > 5):
	nsamp_max = int(sys.argv[5])

if(len(sys.argv) > 6):
	T = np.exp(float(sys.argv[6]))

if(len(sys.argv) > 7):
	w_init = float(sys.argv[7])

if(len(sys.argv) > 8):
	type = sys.argv[8]
	if(type == "ellips"):
		ellips = True
		gasnikov = False
		noisyopt = False
	if(type == "gas"):
		ellips = False
		gasnikov = True
		noisyopt = False
	if(type == "noisyopt"):
		ellips = False
		gasnikov = False
		noisyopt = True

if(len(sys.argv) > 9):
	method = sys.argv[9]


if(not fixed_w):
	outdir = outdir + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i" % (D, w_init, dt, nsamp_max)
else:
	outdir = outdir + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i_fw" % (D, w_init, dt, nsamp_max)



indir = "Ising_test"
if(ellips):
	indir = "Ising_test_ellipsoid" 
	outdir = outdir + "_elps"
	
if(gasnikov):
	indir = "Ising_test_gasnikov" 
	outdir = outdir + "_gas"
	
if(noisyopt):
	indir = "Ising_test_noisyopt" 
	outdir = outdir + "_noisyopt_" + method
	
outdir = outdir + "/N_%i_T_%f_beta_%f" % (N,T,temp_beta)



if(not fixed_w):
	indir = indir + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i" % (D, w_init, dt, nsamp_max)
else:
	indir = indir + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i_fw" % (D, w_init, dt, nsamp_max)

if(noisyopt):
	indir = indir + "_" + method

indir = indir + "/N_%i_T_%f_beta_%f" % (N,T,temp_beta)



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

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

PARAM_NAMES = ["dt", "p_init","p_end","beta"]


param_labels = PARAM_NAMES
for t, xw in zip(t_list, xw_list):
	
	for i in range(D):
		
		plt.plot(t, xw[:,i], label = param_labels[i], color = colors[i])
		
	param_labels = [None]*D

plt.xlabel("number of samples")
plt.ylabel("parameter")
plt.legend()
show_plot()



param_labels = PARAM_NAMES
for t, xw in zip(t_list, xw_list):
	
	for i in range(D):
		
		plt.plot(t, xw[:,i], label = param_labels[i], color = colors[i])
		
	param_labels = [None]*D

plt.xscale("log")
plt.xlabel("number of samples")
plt.ylabel("parameter")
plt.legend()
show_plot()


for t, f in zip(t_list, f_list):
	plt.plot(t, f)

plt.xlabel("number of samples")
plt.ylabel("fitness")
show_plot()


param_labels = PARAM_NAMES
for t, w, xw in zip(t_list, w_list, xw_list):
	
	for i in range(D):
		
		plt.plot(t, xw[:,i], label = param_labels[i], color = colors[i])
	
	
	plt.plot(t, w, color = "gray", dashes = [3,3])	
	param_labels = [None]*D



plt.xlabel("number of samples")
plt.ylabel("parameter")
plt.legend()
show_plot()


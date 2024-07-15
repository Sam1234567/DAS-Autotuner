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
D = 2

if(len(sys.argv) > 1):
	D = int(sys.argv[1])

#no noise for now

#xw_init = np.array([2.0,2.0] + [2.0]*(D-2))
w_init = 0.25


func_idx = 12
noise_type = 1


fixed_w = False

g = 0.05
g_exp = 0.0

sigma = 0.1

kappa = 0.5

table_version = 0

#outpath
outdir = sys.argv[0].split(".")[0]

if(len(sys.argv) > 2):
	func_idx = int(sys.argv[2])
	
if(len(sys.argv) > 3):
	table_version = int(sys.argv[3])	

if(len(sys.argv) > 4):
	g = float(sys.argv[4])
	



outdir = outdir + "/D_%i_fidx_%i" % (D, func_idx)


indir_list = []
label_list = []
	
indir = "convergence_test_ellipsoid"

if(not fixed_w):
	indir = indir + "/D_%i_sigma_%f_w_init_%f_func_%i" % (D, sigma, w_init, func_idx)
else:
	indir = indir + "/D_%i_sigma_%f_w_init_%f_func_%i_fw" % (D, sigma, w_init, func_idx)

if(g > 0):
	indir = indir + "_g_%f_%f" % (g, g_exp)

if(kappa != 1.0):
	indir = indir + "_kappa_%f" % (kappa)


if(noise_type > 0):
	indir = indir + "_noise_%i" % noise_type


indir_list.append(indir)
label_list.append("DAS (this work)")


#for w_init in [0.25, 0.5, 1.0, 2.0]:
for w_init in [0.0, 0.25]:
#for w_init in [0.0, 0.5]:
	indir = "convergence_test_gasnikov"

	if(not fixed_w):
		indir = indir + "/D_%i_sigma_%f_w_init_%f_func_%i" % (D, sigma, w_init, func_idx)
	else:
		indir = indir + "/D_%i_sigma_%f_w_init_%f_func_%i_fw" % (D, sigma, w_init, func_idx)

	if(noise_type > 0):
		indir = indir + "_noise_%i" % noise_type
	
	indir_list.append(indir)
	label_list.append("Gasnikov, w=" + str(w_init))


indir = "convergence_test_noisyopt"

method = "SPSA"
w_init = 2.0

indir = indir + "/D_%i_sigma_%f_w_init_%f_func_%i_%s" % (D, sigma, w_init, func_idx, method)

if(noise_type > 0):
	indir = indir + "_noise_%i" % noise_type


indir_list.append(indir)
label_list.append("SPSA (Spall 1998)")


indir = "convergence_test_bohb"


w_init = 2.0

indir = indir + "/D_%i_sigma_%f_w_init_%f_func_%i" % (D, sigma, w_init, func_idx)

if(noise_type > 0):
	indir = indir + "_noise_%i" % noise_type


indir_list.append(indir)
label_list.append("BOHB (Falkner 2018)")



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


t_range = 10**np.arange(1,6,0.1)


def get_t_f(indir):
	xw_list = []
	t_list = []
	f_list = []
	w_list = []
	
	
	dat_idx = 0
	
	while(os.path.exists("data/" + indir + "/xw%i.txt" % dat_idx)):
	
		xw_list.append(np.loadtxt("data/" + indir + "/xw%i.txt" % dat_idx))
		t_list.append(np.loadtxt("data/" + indir + "/samps_f_w%i.txt" % dat_idx)[:,0])
		f_list.append(np.loadtxt("data/" + indir + "/samps_f_w%i.txt" % dat_idx)[:,1])
		w_list.append(np.loadtxt("data/" + indir + "/samps_f_w%i.txt" % dat_idx)[:,2])
	
		dat_idx += 1

	print("data/" + indir + "/xw%i.txt" % dat_idx, dat_idx)
	
	average_f = np.zeros(len(t_range))

	for i in range(0, len(t_list)):
		average_f += np.interp(t_range, t_list[i], f_list[i])
	average_f = average_f/len(t_list)
	return t_list, f_list, average_f, w_list, xw_list


t_list_list = []
f_list_list = []

for indir in indir_list:
	t_list, f_list, average_f, w_list, xw_list = get_t_f(indir)
	t_list_list.append(t_list)
	f_list_list.append(f_list)
	
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
f_opt = 1

t_range = 10**np.arange(0,6.5,0.02)
for t_list, f_list, label, color in zip(t_list_list, f_list_list, label_list, colors):
	
	f_avg  = np.zeros(len(t_range))
	
	for t, f in zip(t_list, f_list):
		plt.plot(t, f_opt - f,   color = color, alpha = 0.2)
		f_avg += np.interp(t_range, t, f)
	
	plt.plot(t_range, f_opt - f_avg/len(t_list),   color = color, label = label)
	


plt.xlim((1, 10**6.5))
plt.yscale("log")
plt.xscale("log")
plt.xlabel("number of samples")
plt.ylabel(r"$f_{opt} - f(x)$")
plt.legend(fontsize = fontsize)
show_plot()
plt.close()



t_range = 10**np.arange(0,6.5,0.02)
for t_list, f_list, label, color in zip(t_list_list, f_list_list, label_list, colors):
	
	f_avg  = np.zeros(len(t_range))
	f_var = np.zeros(len(t_range))
	f_max = -100*np.ones(len(t_range))
	f_min = 100*np.ones(len(t_range))
	for t, f in zip(t_list, f_list):
		#plt.plot(t, f,   color = color, alpha = 0.2)
		f_interp = np.interp(t_range, t, f)
		f_avg += f_interp
		f_var += f_interp**2
		f_max = np.maximum(f_interp, f_max)
		f_min = np.minimum(f_interp, f_min)
	#print(label, f_avg)
	
	
	f_avg = f_avg/len(t_list)
	f_var = f_var/len(t_list) - f_avg**2
	if(len(t_list) > 0):
		plt.fill_between(t_range,  f_avg - np.sqrt(f_var),  f_avg + np.sqrt(f_var), color = color, alpha = 0.2)
		plt.plot(t_range,  f_avg, color = color, label = label)


plt.xlim((10**3.5, 10**6.5))
plt.ylim((0.0, 1.0))
#plt.yscale("log")
plt.xscale("log")
plt.xlabel("number of samples")
plt.ylabel(r"$f_{opt} - f(x)$")

plt.ylabel(r"$f(x)$")
plt.legend(fontsize = fontsize)
show_plot()
plt.close()



t_range = np.arange(0, 10**5, 40)
for t_list, f_list, label, color in zip(t_list_list, f_list_list, label_list, colors):
	
	f_avg  = np.zeros(len(t_range))
	
	for t, f in zip(t_list, f_list):
		plt.plot(t, f,   color = color, alpha = 0.2)
		f_avg += np.interp(t_range, t, f)
	print(label, f_avg)
	plt.plot(t_range,  f_avg/len(t_list),   color = color, label = label)
	




plt.xlim((0, 10**5))
plt.xlabel("number of samples")
plt.ylabel(r"$f(x)$")
plt.legend(fontsize = fontsize)
show_plot()
plt.close()



t_range = np.arange(0, 10**5, 40)
for t_list, f_list, label, color in zip(t_list_list, f_list_list, label_list, colors):
	
	f_avg = np.zeros(len(t_range))
	f_var = np.zeros(len(t_range))
	
	f_max = -100*np.ones(len(t_range))
	f_min = 100*np.ones(len(t_range))
	for t, f in zip(t_list, f_list):
		#plt.plot(t, f,   color = color, alpha = 0.2)
		f_interp = np.interp(t_range, t, f)
		f_avg += f_interp
		f_var += f_interp**2
		f_max = np.maximum(f_interp, f_max)
		f_min = np.minimum(f_interp, f_min)
	#print(label, f_avg)
	
	f_avg = f_avg/len(t_list)
	f_var = f_var/len(t_list) - f_avg**2
	if(len(t_list) > 0):
		plt.fill_between(t_range,  f_avg - np.sqrt(f_var),  f_avg + np.sqrt(f_var), color = color, alpha = 0.2)
		plt.plot(t_range,  f_avg, color = color, label = label)




plt.xlim((0, 10**5))
plt.xlabel("number of samples")
plt.ylabel(r"$f(x)$")
plt.legend(fontsize = fontsize)
show_plot()
plt.close()



#print tex for table
ns_list = [10**3, 10**4, 10**5]
columns = ["$n_s = 10^3$", "$n_s = 10^4$", "$n_s = 10^5$"]

if(table_version == 1):
	
	ns_list = [10**4, 10**5, 10**6]
	columns = ["$n_s = 10^4$", "$n_s = 10^5$", "$n_s = 10^6$"]


out = "\nlatex table: \n\n"

for col in columns:
	out = out + "& %s & %s & %s" % (col + " avg", col + " min", col + " max")
	 
print(out + " \\\\ \hline")

column_list = []
row_labels = []

row_idx = 0
for t_list, f_list, label, color in zip(t_list_list, f_list_list, label_list, colors):
	
	
	
	
	t_range = np.array(ns_list)
	f_interp = np.interp(t_range, t, f)
	
		
	f_avg  = np.zeros(len(t_range))
	
	f_max = -100*np.ones(len(t_range))
	f_min = 100*np.ones(len(t_range))
	for t, f in zip(t_list, f_list):
		#plt.plot(t, f,   color = color, alpha = 0.2)
		f_interp = np.interp(t_range, t, f)
		f_avg += f_interp
		
		f_max = np.maximum(f_interp, f_max)
		f_min = np.minimum(f_interp, f_min)
	if(len(t_list) > 0):
		row_labels.append(label)
		f_avg = f_avg / len(t_list)
		col_idx = 0
		for col, favg, fmin, fmax in zip(columns, f_avg, f_min, f_max):
			
			if(len(column_list) <= col_idx):
				column_list = column_list + [[],[],[]]
			
			
			column_list[col_idx].append(favg)
			column_list[col_idx+1].append(fmin)
			column_list[col_idx+2].append(fmax)
			col_idx += 3
	
	row_idx += 1
		
for row_idx in range(len(column_list[0])):
	
	out = row_labels[row_idx] + " "
	for col_idx in range(len(column_list)):
		if(np.argmax(column_list[col_idx]) == row_idx):
			out = out + " & \\textbf{%.3f}" % (column_list[col_idx][row_idx])
		else:
			out = out + " & %.3f" % (column_list[col_idx][row_idx])
	
		
	
	print(out + " \\\\ ")
	
		
		
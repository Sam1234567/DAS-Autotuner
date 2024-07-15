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
D = 2

if(len(sys.argv) > 1):
	D = int(sys.argv[1])

#no noise for now

#xw_init = np.array([2.0,2.0] + [2.0]*(D-2))
w_init = 2.0

g = 0.25
g_exp = 0.25

func_idx = 12
noise_type = 1


fixed_w = False

sigma = 0.1

#outpath
outdir = sys.argv[0].split(".")[0]





outdir = outdir + "/D_%i_fidx_%i" % (D, func_idx)



indir_1 = "convergence_test_ellipsoid"

if(not fixed_w):
	indir_1 = indir_1 + "/D_%i_sigma_%f_w_init_%f_func_%i" % (D, sigma, w_init, func_idx)
else:
	indir_1 = indir_1 + "/D_%i_sigma_%f_w_init_%f_func_%i_fw" % (D, sigma, w_init, func_idx)

if(noise_type > 0):
	indir_1 = indir_1 + "_noise_%i" % noise_type


indir_2 = "convergence_test_ellipsoid"

if(not fixed_w):
	indir_2 = indir_2 + "/D_%i_sigma_%f_w_init_%f_func_%i" % (D, sigma, w_init, func_idx)
else:
	indir_2 = indir_2 + "/D_%i_sigma_%f_w_init_%f_func_%i_fw" % (D, sigma, w_init, func_idx)

if(g > 0):
	indir_2 = indir_2 + "_g_%f_%f" % (g, g_exp)


if(noise_type > 0):
	indir_2 = indir_2 + "_noise_%i" % noise_type





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

	
	
	average_f = np.zeros(len(t_range))

	for i in range(0, len(t_list)):
		average_f += np.interp(t_range, t_list[i], f_list[i])
	average_f = average_f/len(t_list)
	return t_list, f_list, average_f, w_list, xw_list


t_list_1, f_list_1, average_f_1, w_list_1, xw_list_1 = get_t_f(indir_1)


t_list_2, f_list_2, average_f_2, w_list_2, xw_list_2 = get_t_f(indir_2)





	
if(len(t_list_1) > 0):
	plt.plot(t_list_1[0], 1 - f_list_1[0], color = "blue", alpha = 0.2, label = "No Limit")
	for i in range(1, len(t_list_1)):
		plt.plot(t_list_1[i], 1 - f_list_1[i], alpha = 0.2,color = "blue")

	plt.plot(t_range, 1 - average_f_1, label = "No Limit (Average)")

if(len(t_list_2) > 0):
	plt.plot(t_list_2[0], 1 - f_list_2[0], color = "red", alpha = 0.2, label = "Limited Shrinking")
	for i in range(1, len(t_list_2)):
		plt.plot(t_list_2[i], 1 - f_list_2[i], alpha = 0.2,color = "red")

	plt.plot(t_range, 1 - average_f_2, label = "Limited Shrinking (average)")



plt.yscale("log")
plt.xscale("log")
plt.xlabel("number of samples")
plt.ylabel(r"$f_{opt} - f(x)$")
plt.legend()
show_plot()
plt.close()



L_list_2 = []

L_list_2 = np.loadtxt("data/" + indir_2 + "/L%i.txt" % 0)

circlex = np.cos(np.arange(0,2,0.01)*np.pi)
circley = np.sin(np.arange(0,2,0.01)*np.pi)



f = toy_funcs.get_func(func_idx)

#display fitness function
z_rng = np.arange(-2,2,0.02)
X_,Y_ = np.meshgrid(z_rng,z_rng)

z = np.zeros((D, len(z_rng), len(z_rng)))
z[0,:,:] = X_
z[1,:,:] = Y_
F_ = f(z)

#plt.title("fitness function")
plt.xlim(-2,2)
plt.ylim(-2,2)

plt.pcolormesh(X_,Y_, F_)

run_idx = 1
xw = xw_list_1[run_idx]
for  idx, w in enumerate(w_list_1[run_idx]):
	
	
	other_dims = [run_idx]*(D-2)
	print(xw[idx])
	boundxy = [xw[idx] + w*np.array([x,y] + list(other_dims)) for x,y in zip(circlex, circley)]
	#boundxy = [[x,y] for x,y in zip(circlex, circley)]
	plt.plot(np.array([xy[0] for xy in boundxy]), np.array([xy[1] for xy in boundxy]), color = "red", alpha = 0.5)
	
	

plt.plot([],[], color = "red", label = "sampling window")
plt.plot([1],[1], marker = "o", color = "blue", label = "optimum")


plt.colorbar()
plt.legend()

show_plot()
plt.close()


plt.xlim(-2,2)
plt.ylim(-2,2)


plt.pcolormesh(X_,Y_, F_)


run_idx = 0

xw = xw_list_2[run_idx]
for  idx, L in enumerate(L_list_2):
	L = L.reshape(D,D)
	xw_idx = int(idx*len(xw)/len(L_list_2))
	
	
	other_dims = [0]*(D-2)
	boundxy = [xw[xw_idx] + np.dot(L, np.array([x,y] + list(other_dims))) for x,y in zip(circlex, circley)]
	#boundxy = [[x,y] for x,y in zip(circlex, circley)]
	plt.plot(np.array([xy[0] for xy in boundxy]), np.array([xy[1] for xy in boundxy]), color = "red", alpha = 0.5)
	
	
plt.plot([],[], color = "red", label = "sampling window")
plt.plot([1],[1], marker = "o", color = "blue", label = "optimum")


plt.colorbar()
plt.legend()

show_plot()
plt.close()

plt.plot(t_list_1[run_idx], [0]*len(t_list_1[run_idx]), dashes = [1,1], color = "gray")
run_idx = 1
plt.plot(t_list_1[run_idx], [xw[0] for xw in xw_list_1[run_idx]] ,color = "red", label = "x-coord, circular")
plt.plot(t_list_1[run_idx], [xw[1] for xw in xw_list_1[run_idx]] ,color = "blue", label = "x-coord, circular")
plt.plot(t_list_1[run_idx], w_list_1[run_idx] ,color = "green", label = "window size, circular")

run_idx = 2
plt.plot(t_list_2[run_idx], [xw[0] for xw in xw_list_2[run_idx]] ,color = "red", dashes = [3,3], label = "x-coord, elliptical")
plt.plot(t_list_2[run_idx], [xw[1] for xw in xw_list_2[run_idx]] ,color = "blue", dashes = [3,3], label = "x-coord, elliptical")
plt.plot(t_list_2[run_idx], w_list_2[run_idx] ,color = "green", dashes = [3,3], label = "window size, elliptical")


plt.ylabel("coordinate")
plt.xlabel("number of samples")
plt.xscale("log")
plt.legend()
show_plot()
plt.close()


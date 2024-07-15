#this code tests the tuner on a specific SAT instance 

import numpy as np
import matplotlib.pyplot as plt

import SGD_sampler_ellipsoid as samp
import toy_funcs

import noisyopt

import CIM_CAC as cimcac
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
xw_init = 2*np.random.rand(D)
w_init = 1.0



func_idx = 5

fixed_w = False

N = 150

temp_beta = 0.01
# temp_beta = 0.01
method = "SPSA"

dt = 0.1
nsamp_max = 100000


T = np.exp(5.0)



# 4 150 0.01 0.1 50000 5.0 2.0 SPSA
# 4 300 0.005 0.1 50000 5.0 2.0 SPSA
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
	method = sys.argv[8]


#outpath
outdir = sys.argv[0].split(".")[0]




if(not fixed_w):
	outdir = outdir + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i" % (D, w_init, dt, nsamp_max)
else:
	outdir = outdir + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i_fw" % (D, w_init, dt, nsamp_max)

outdir = outdir + "_" + str(method)


outdir = outdir + "/N_%i_T_%f_beta_%f" % (N,T, temp_beta)

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









xw = xw_init*1
w = w_init*1

def sample(x):
	R = x.shape[1]
	print(R)
	
	SEED = int(np.random.rand()*1000)
	
	random.seed(SEED)

	#generate problem instance
	J = inst_uils.randGaussianSK(N)
	


	#setup solver
	pt_device = "cpu"
	
	solver = cimcac.CIM(pt_device, N, J)
	solver.norm = 1/np.sqrt(N)
	
	#set parameters
	solver.T = T
	
	PARAM_NAMES = ["dt", "p_init","p_end","beta"]
	
	#PARAM_NAMES = ["dt", "p_init", "p_end"]
	#PARAM_NAMES = ["dt", "p_init"]
	
	for idx, param_name in enumerate(PARAM_NAMES):
		setattr(solver, param_name, x[idx, :])
	
	# cac.Dt = np.maximum(0.0, cacm.Dt)
# 	cac.xi = np.maximum(0.0, cacm.nl_sat)

	#initialize solver to run 10 trajectories

	solver.init(R)

	#solve
	print("solving")

	tstart = time.time()
	Ps, E_opt = solver.traj(0)
	print(E_opt)
	
	E_approx = (-0.761 + 0.7 * N**(-2/3))*N**(3/2)
	#print(E_approx)
	
	fitness = (np.exp(-temp_beta*(E_opt - E_approx)))
	print(np.average(fitness))
	return fitness

def f(x):
	R_eval = 50
	x_extended = np.outer(x, np.ones(R_eval))
	N_inst = 20
		
	return np.average([np.average(sample(x_extended)) for i in range(N_inst)])
	
	

#use stochastic gradient descent sampler	

tot_samp_rec = []
xw_rec = []
w_rec = []
count = 0
R_eval = 10

def obj(x):
	global count
	x = x
	count += 1
	N_inst = 1
	
	if(count % 10 == 0):
		tot_samp_rec.append(count*N_inst*R_eval)
		xw_rec.append(x)
		w_rec.append(1)
	
	print(count, x)
	
	
	x_extended = np.outer(x, np.ones(R_eval))
	
		
	return -np.average([np.average(sample(x_extended)) for i in range(N_inst)])
	
print("calculating sampled traj...")

if(method == "SPSA"):
	res, x_list = noisyopt.minimizeSPSA(obj, x0=xw_init, niter = int(nsamp_max/R_eval),  c=w_init, a = dt, alpha = 0.6, gamma= 0.3, paired=False)
if(method == "DS"):
	res, x_list = noisyopt.minimizeCompass(obj, x0=xw_init, niter = nsamp_max/R_eval, errorcontrol = True, deltatol=0.05, feps = 0.01, paired=False)

print(x_list)
xw_rec = x_list
tot_samp_rec = (1 + np.array(range(len(x_list))))*2*50*R_eval
w_rec = [1]*len(xw_rec)

# res = noisyopt.minimizeCompass(obj, x0=xw_init, deltatol=0.05, paired=False)
# print(res)
# 
# res = noisyopt.minimizeSPSA(obj, x0=xw_init, niter = 5000, a=0.1, c=0.1, paired=False)
# print(res)





#every "stride" steps we re-evaluate f to get an idea of fitness over time

stride = 1

xw_rec_reduced = np.array(xw_rec)[::stride,:]
tot_samp_rec_reduced = np.array(tot_samp_rec)[::stride]
w_rec_reduced = np.array(w_rec)[::stride]

print("estimating true fitness function...")
f_list = [f(x) for x in xw_rec_reduced]
print(f_list)

save_data("samps_f_w", np.array([tot_samp_rec_reduced, f_list, w_rec_reduced]).T)
save_data("xw", xw_rec_reduced)




#plots

x_pos = [xw[0] for xw in xw_rec]
x2_pos = [xw[1] for xw in xw_rec]

plt.xlabel("$x_0$")
plt.ylabel("$x_1$")


plt.plot(x_pos, x2_pos, label = "sampler trajectory")

plt.legend()
show_plot()
plt.close()



plt.xlabel("number of samples")
plt.ylabel("fitness")
plt.plot(tot_samp_rec_reduced, f_list)
show_plot()
plt.close()


plt.xlabel("number of samples")
plt.ylabel("$1-f(x_w)$")

plt.xscale("log")
plt.yscale("log")

for c in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
	plt.plot(tot_samp_rec_reduced, np.array(tot_samp_rec_reduced)**(-0.5)*c, color = "gray", dashes = [3,3])


plt.plot(tot_samp_rec_reduced, 1-np.array(f_list))
show_plot()
plt.close()

plt.xlabel("number of samples")
plt.ylabel("$w$")
plt.plot(tot_samp_rec_reduced, w_rec_reduced)
show_plot()
plt.close()


plt.xlabel("number of samples")
plt.ylabel("$w$")
plt.xscale("log")
plt.yscale("log")

plt.plot(tot_samp_rec_reduced, w_rec_reduced)
show_plot()
plt.close()



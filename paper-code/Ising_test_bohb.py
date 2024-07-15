#this code tests the tuner on a specific SAT instance 

import numpy as np
import matplotlib.pyplot as plt

import SGD_sampler_ellipsoid as samp
import toy_funcs

import CIM_CAC as cimcac
import InstanceUtils as inst_uils
import random

import logging
logging.basicConfig(level=logging.WARNING)


import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker

import ConfigSpace as CS
from hpbandster.core.worker import Worker


import numpy as np
import noisyopt 


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
xw_init = np.random.rand(D)
w_init = 0.5




fixed_w = False

N = 150


dt = 0.5
nsamp_max = 50000


T = np.exp(5.0)


# 4 150 0.01 50000 5.0 2.0
# 4 300 0.005 50000 5.0 2.0
if(len(sys.argv) > 2):
	N = int(sys.argv[2])

if(len(sys.argv) > 3):
	temp_beta = float(sys.argv[3])


if(len(sys.argv) > 4):
	nsamp_max = int(sys.argv[4])

if(len(sys.argv) > 5):
	T = np.exp(float(sys.argv[5]))

if(len(sys.argv) > 6):
	w_init = float(sys.argv[6])




#outpath
outdir = sys.argv[0].split(".")[0]






if(not fixed_w):
	outdir = outdir + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i" % (D, w_init, dt, nsamp_max)
else:
	outdir = outdir + "/D_%i_w_init_%f_dt_%f_nsamp_max_%i_fw" % (D, w_init, dt, nsamp_max)

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









xw = xw_init*1.0
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
	
	print(E_approx)
	
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


def obj(x, R):
	global count
	x = x
	count += 1
	N_inst = 1
	
	
	print(count, x)
	
	
	x_extended = np.outer(x, np.ones(R))
	
		
	return -np.average([np.average(sample(x_extended)) for i in range(N_inst)])
	


tot_runs = 0
x_list = []
tot_samp_rec = []
res_best = 10**10
best_x = None

class CustomWorkser(Worker):
	
	
	def __init__(self, *args, sleep_interval=0, **kwargs):
		
		super().__init__(*args, **kwargs)

		self.sleep_interval = sleep_interval

	def compute(self, config, budget, **kwargs):
		global tot_runs, res_best, best_x, x_list, tot_samp_rec
		
		tot_runs += int(budget)
		
		
		x = np.zeros(D)
		
		for i in range(D):
			x[i] = config["x" + str(i)]
		
		res = obj(x, int(budget))
		
		if(res + 1/budget <= res_best):
			res_best = res + 1/budget
			best_x = x
		
		x_list.append(best_x)
		tot_samp_rec.append(tot_runs)
		
		
		
		#print(time.time() - tstart, "s")
		print("loss", float(res))
		return({
					'loss': float(res),
					'info': {}  # this is the a mandatory field to run hyperband
				})
	
	
	@staticmethod
	def get_configspace():
		config_space = CS.ConfigurationSpace()
			
		for i in range(D):
			config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x' + str(i), lower= 0, upper= w_init))
		
		return(config_space)





host = '127.0.0.1'

NS = hpns.NameServer(run_id='example1', host=host, port=None)
NS.start()

for i in range(1):
	w = CustomWorkser(sleep_interval = 0, nameserver=host,run_id='example1', id = i)
	
	w.run(background=True)
print("here")


n_it = 150
max_bud = 100
#previous_run
bohb = BOHB(  configspace = w.get_configspace(),
			  run_id = 'example1', nameserver=host,
			  min_budget=4, max_budget=max_bud)
			  #,
			 #  previous_result = previous_run)
res = bohb.run(n_iterations = n_it)

TR = tot_runs


#print(bohb.config_generator.kde_models[list(kys)[0]]['good'].pdf([0.5 , 100]))

previous_run = res
# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

best_config = id2config[incumbent]['config']
print('Best found configuration:', id2config[incumbent])
print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))

params_center = best_config

all_runs = res.get_all_runs()

losses = [r.loss for r in all_runs]
print(losses)
loss_ratio = np.max(losses)/np.min(losses)
print("loss ratio: " , loss_ratio)



#every "stride" steps we re-evaluate f to get an idea of fitness over time

stride = 10


xw_rec = x_list

w_rec = [1]*(len(x_list))

print(len(xw_rec), len(tot_samp_rec), len(w_rec))


xw_rec_reduced = np.array(xw_rec)[::stride,:]
tot_samp_rec_reduced = np.array(tot_samp_rec)[::stride]
w_rec_reduced = np.array(w_rec)[::stride]

print("estimating true fitness function...")
f_list = [f(x) for x in xw_rec_reduced]
print(f_list)
print(len(tot_samp_rec_reduced), len(f_list), len(w_rec_reduced))


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



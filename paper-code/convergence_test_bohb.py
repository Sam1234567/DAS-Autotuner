#this code checks the asymptotic scaling of the error with respect to number of samples

import numpy as np
import matplotlib.pyplot as plt

import SGD_sampler as samp
import toy_funcs

import logging
logging.basicConfig(level=logging.WARNING)


import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker

import ConfigSpace as CS
from hpbandster.core.worker import Worker

import time

#import noisyopt 

import noisyopt


import sys
import os



#script parameters
D = 4

if(len(sys.argv) > 1):
	D = int(sys.argv[1])

#no noise for now
sigma = 0.1


#xw_init = np.array([2.0,2.0] + [2.0]*(D-2))
xw_init = np.array([2.33]*D)
xw_init = np.array([-1.2,1.0]*int(D/2))
xw_init = np.array([0.0, 0.5]*int(D/2))
w_init = 2.0



func_idx = 12
noise_type = 1


init_mode = -1

#2 12 0.1 1 0 40000000

if(len(sys.argv) > 2):
	func_idx = int(sys.argv[2])
	
if(len(sys.argv) > 3):
	sigma = float(sys.argv[3])

if(len(sys.argv) > 4):
	noise_type = int(sys.argv[4])

if(len(sys.argv) > 6):
	nsamp_max = int(sys.argv[6])


#outpath
outdir = sys.argv[0].split(".")[0]



outdir = outdir + "/D_%i_sigma_%f_w_init_%f_func_%i" % (D, sigma, w_init, func_idx)

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

def obj(x, R):
	global eval_count
	eval_count += 1
	
	if(eval_count % 100 == 0):
		#tot_samp_rec.append(eval_count*R)
		xw_rec.append(x)
		w_rec.append(1)
	if(eval_count % 100 == 0):
		print(eval_count, f(x), x)
	x_extended = np.outer(x, np.ones(R))
	return -np.average(sample(x_extended))





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
		
		tot_runs += budget
		
		
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
			config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x' + str(i), lower= 0, upper= 1))
		
		return(config_space)





host = '127.0.0.1'

NS = hpns.NameServer(run_id='example1', host=host, port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.

for i in range(1):
	w = CustomWorkser(sleep_interval = 0, nameserver=host,run_id='example1', id = i)
	
	w.run(background=True)
print("here")


# for i in range(n_workers):
# 			workers[i].prob = prob
# 			workers[i].E0 = E0
# 		
# if(len(res_all) > 0):
# 	out = w.compute(res_all[len(res_all)-1], 100)
# 	prev_loss_list.append(out['loss'])
# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run BOHB, but that is not essential.
# The run method will return the `Result` that contains all runs performed.
n_it = 500
max_bud = 40
#previous_run
bohb = BOHB(  configspace = w.get_configspace(),
			  run_id = 'example1', nameserver=host,
			  min_budget=4, max_budget=max_bud)
			  #,
			 #  previous_result = previous_run)
res = bohb.run(n_iterations= n_it)

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



#print(x_list)
xw_rec = x_list
w_rec = [1]*len(xw_rec)

f_list = [f(x) for x in xw_rec]
print(len(xw_rec), len(tot_samp_rec))

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



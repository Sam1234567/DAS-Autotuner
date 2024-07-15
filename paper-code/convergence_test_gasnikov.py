#this code checks the asymptotic scaling of the error with respect to number of samples

import numpy as np
import matplotlib.pyplot as plt

import SGD_sampler as samp
import toy_funcs

import tensorflow as tf

#import noisyopt 

import noisyopt


import sys
import os



#script parameters
D = 8





#no noise for now
sigma = 0.1


#xw_init = np.array([2.0,2.0] + [2.0]*(D-2))
xw_init = np.array([2.33]*D)
#xw_init = np.array([-1.2,1.0]*int(D/2))
xw_init = np.array([0.0, 0.5]*int(D/2))
xw_init = np.random.rand(D)

w_init = 0.5

if(len(sys.argv) > 1):
	D = int(sys.argv[1])

if(len(sys.argv) > 2):
	w_init = float(sys.argv[2])
	print(w_init)



func_idx = 15
noise_type = 1

#method = "SPSA"
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
count = 0


def obj(x):
	global count
	x = x
	count += 1
	
	R_eval = 1
	if(count % 10 == 0):
		tot_samp_rec.append(count*R_eval)
		xw_rec.append(x)
		w_rec.append(1)
	
	print(count, x)
	
	
	x_extended = np.outer(x, np.ones(R_eval))
	N_inst = 1
		
	return -np.average([np.average(sample(x_extended)) for i in range(N_inst)])
	
print("calculating sampled traj...")

B = 10


xw = xw_init + 0


optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9,
    beta_2=0.999, epsilon=1e-07,)

#optimizer = tf.keras.optimizers.SGD(learning_rate=0.3, momentum = 0.99)


x = tf.Variable(xw, dtype = np.float64)

nsamp_max = 200000


for i in range(20000):
	v = np.random.randn(D, B)
	v = v/np.linalg.norm(v, axis = 0).reshape(1,B)
	v = v*w_init
	
	
	#print(xw + v, xw - v)
	if(i % 50 == 0):
		tot_samp_rec.append(B*2*i)
		xw_rec.append(xw)
		w_rec.append(w)
	v_ = np.zeros((D, B*2))
	v_[:, :B] = v
	v_[:, B:] = -v
	#print(np.outer(v, vec))
	g = (np.average(v_*sample(xw.reshape(D,1) + v_).reshape(1,2*B), axis = 1))/(w_init)
	print(xw, g)
	
	optimizer.apply_gradients([(-g, x)])
	xw = np.array(x)
	
	if(B*2*i > nsamp_max):
		break
	
	
	
	#xw = xw + dt*g*w_init**2



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



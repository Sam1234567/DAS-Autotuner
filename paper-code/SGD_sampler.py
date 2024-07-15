#This is the main code for the tuner (DIS)
#It uses gradient descent to optimize window size, shape and position simultaneously.
#Class take in sample function and dimension D.
#no momentum or previous samples used
import numpy as np



class Sampler:
	
	
	
	def __init__(self, sample, D):
		self.D = D
		self.sample = lambda  x : sample(x)
		
		
		self.xw = np.zeros(D)
		self.w = 1
		
		self.tot_samp  = 0
		self.dt = 0.1
		
		self.fixed_w = False
	
	
	def init_window(self, xw, w):
		self.xw = xw
		self.w = w
	
	def step(self, dt, B):
		self.tot_samp += B
		D = self.D
		#random samples (z = normalized coord, x = real coord)
		z_samp = np.random.randn(D, B)
		x_samp = self.xw.reshape(D, 1) + self.w*z_samp
		#sample fitness function
		y_samp = self.sample(x_samp)
		
		#differentials in normalized coordinates
		dz = np.average(z_samp*y_samp.reshape(1,B), axis = 1)
		dw = -D*np.average(y_samp) + np.average(np.sum(z_samp**2, axis = 0)*y_samp)
		
		if(self.fixed_w):
			dw = 0
		
		dxw =  dz/self.w**1
		
		dw = dw/(self.w**1)/(2*D)
		
		#additional factor r is used to help with numerical stability when updating w. 
		#This steps keeps the window from shrinking too quickly.
		used_dt = np.minimum(dt, self.w**2*dt)
		
		#fixed w uses a different time step (for comparison purposes onnly)
		if(self.fixed_w):
			used_dt = dt/(self.numb_steps + 1)**0.5
		
		w_ = self.w + used_dt*dw
		r = (w_/self.w)**2
		r = 1
		self.w = self.w + r*used_dt*dw
		
		
		#update xw
		self.xw = self.xw + r*used_dt*dxw
		
		
		self.numb_steps += 1
		
	
	def optimize(self, tot_samp_max = 5000, tr_min = 0):
		self.tot_samp  =0
		self.numb_steps = 0
		tot_samp_rec = []
		xw_rec = []
		w_rec = []
		
		count = 0
		while(self.tot_samp < tot_samp_max and np.abs(self.w) > tr_min):
			#batch size chosen as 1/trace(L) here so more accuracy when window shrinks
			B = int(10/self.w)
			
			#print (for debug)
			if(True and count % 20 == 0):
				print(B)
				print(self.xw)
				
				
			self.step(self.dt, B)
			
			#save info
			tot_samp_rec.append(self.tot_samp)
			xw_rec.append(self.xw)
			w_rec.append(self.w)
			count += 1
		
		return 	tot_samp_rec, xw_rec, w_rec
	
	
	
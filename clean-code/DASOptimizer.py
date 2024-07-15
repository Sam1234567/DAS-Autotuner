#Updated DAS Optimizer code
#Based on algorithm described in "Dynamic Anisotropic Smoothing for Noisy Derivative-Free Optimization" but with a few changes/

#1: Time step for SDE is set to ensure stability. Only initial rough estimate of objective optimum is needed
#2: Batch size is fixed (not varying)
#3: Diagonal Hessian estimation (instead of full Hessian estimation) is supported for high dimensional problems such as NNs

import numpy as np



class DAS:
	
	#hyperparameters
	
	eps = 10**(-10)
	#background growth rate of dt
	rms_beta = 0.99
	#smoothing coefficient for exponential moving average of fit estimate
	fit_est_beta = 0.05
	#normalized time step
	dt0 = 0.5
	#batch size
	B = 100
	#max size of window
	max_win = 2
	
	#diagonal hessian?
	diag = False
	
	
	prev_xw = []
	prev_L = []
	prev_xsamp = []
	prev_ysamp = []
	
	
	
	def __init__(self, sample, D, initial_est = 1.0):
		self.D = D
		self.sample = lambda  x : sample(x)
		
		
		self.fit_est = initial_est
		self.curv_est = 0.000
		self.grad_est = np.zeros(D)
		
		
		self.best_fit2 = initial_est**2
		
	
	
	def init_window(self, xw, L):
		self.xw = xw
		self.L = L
	
	def step(self, B):
		self.tot_samp += B
		D = self.D
		
		#random perturbations (z = normalized coord, x = real coord)
		z_samp = np.random.randn(D, B)
		
		
		if(self.diag):
			x_samp = self.xw.reshape(D, 1) + self.L.reshape(D,1)*z_samp
		else:
			x_samp = self.xw.reshape(D, 1) + np.dot(self.L, z_samp)
		
		
		
		#sample fitness function
		y_samp = self.sample(x_samp)
		
		
		best_fit_ = self.best_fit2 + 0.0
		self.best_fit2 = np.maximum(np.average(y_samp**2), self.best_fit2*self.rms_beta)
		
		#RMS normalization (adaptive time step)
			
		dt = self.dt = self.dt0/np.sqrt(self.best_fit2 + self.eps)
		
		if(self.diag):
			
			#differentials in normalized coordinates
			dz = np.average(z_samp*y_samp.reshape(1,B), axis = 1)
			dA = -np.ones(D)*np.average(y_samp) + np.average(z_samp**2*y_samp.reshape(1,B), axis = 1)
			
			
			#change of basis
			dxw =  self.L*dz*1
			dL =  self.L*dA*1/D
			
		else:
			#differentials in normalized coordinates
			dz = np.average(z_samp*y_samp.reshape(1,B), axis = 1)
			dA = -np.diag(np.ones(D))*np.average(y_samp) + np.average(z_samp.reshape(D, 1, B)*z_samp.reshape(1, D, B)*y_samp.reshape(1, 1,B), axis = 2)
			
			
			#change of basis
			dxw =  np.dot(self.L, dz)*1
			dL =  np.dot(self.L, dA)*1/D
		
		
		### update L
		
		#additional factor r is used to help with numerical stability when updating L. 
		#This steps keeps the window from shrinking too quickly.
		L_ = self.L + dt*dL
		r = np.sum(L_**2)**0.5/np.sum(self.L**2)**0.5
		
		self.L = self.L + r*dt*dL
		
		#max size of window
		
		if(np.sqrt(np.average(self.L**2)) > self.max_win):
			self.L =  self.L*self.max_win/np.sqrt(np.average(self.L**2 ))
		
		
		
		### update x
		
		#ensure x step is not too big
		
		xw_step = r*dt*dxw
		
		if(self.diag):
			zw_step = xw_step / self.L.reshape(D,1)
		else:
			zw_step = np.linalg.solve(self.L, xw_step)
		
		r2 = 1/np.maximum(1, np.linalg.norm(zw_step))
		
		self.xw = self.xw + r2*xw_step
		
		
		#estimate fitness landscape properties (not used in optimization process, but useful metrics)
		
		fest = np.average(y_samp)
		cest = np.average(y_samp*np.average(z_samp**2, axis = 0))
		gest = np.average(y_samp.reshape(1,B)*z_samp.reshape(D,B), axis = 1)
		
		self.fit_est = (1 - self.fit_est_beta)*self.fit_est + self.fit_est_beta*fest
		self.curv_est = (1 - self.fit_est_beta)*self.curv_est + self.fit_est_beta*cest
		self.grad_est = (1 - self.fit_est_beta)*self.grad_est + self.fit_est_beta*gest
		
	
	def optimize(self, tot_samp_max = 50000, verbose = False):
		
		self.tot_samp  =0
		self.tot_samp_max = tot_samp_max
		tot_samp_rec = []
		xw_rec = []
		L_rec = []
		
		count = 0
		#print(self.tot_samp , tot_samp_max, np.abs(np.sum(self.L**2)), (self.curv_est - np.linalg.norm(self.grad_est))/self.fit_est , R_end)
		
		if(verbose):
			print("<step> <tot samp> <fit est> <dt> <window size> <curv est>")
		
		while(self.tot_samp < tot_samp_max):
			
			R = (self.curv_est - np.linalg.norm(self.grad_est))/self.fit_est
			
			self.step(self.B)
			
			#print (for debug)
			if(verbose and count % 50 == 0):
				
				print("x", self.xw)
				
				r = (self.curv_est - np.linalg.norm(self.grad_est))/self.fit_est
				wsize =  np.average(self.L**2)
				
				print(count, self.tot_samp, self.fit_est, self.dt , wsize, r)
				
				
			
			#save info
			tot_samp_rec.append(self.tot_samp)
			xw_rec.append(self.xw)
			L_rec.append(self.L)
			count += 1
		
		
		
		info = {"tot_samp_traj": tot_samp_rec, "xw_traj": xw_rec, "L_rec": xw_rec}
		
		return self.xw, info
	
	
	
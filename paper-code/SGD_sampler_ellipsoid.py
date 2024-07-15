#This is the main code for the tuner (DAS)
#It uses gradient descent to optimize window size, shape and position simultaneously.
#Class takes in sample function and dimension D.
#no momentum or previous samples used
import numpy as np



class Sampler:
	
	
	
	def __init__(self, sample, D):
		self.D = D
		self.sample = lambda  x : sample(x)
		
		
		self.xw = np.zeros(D)
		self.L = np.diag(np.ones(D))
		
		self.tot_samp  = 0
		self.dt = 0.1
		self.g = 0.0
		self.g_exp = 0.5
		self.kappa = 1.0
	
	
	def init_window(self, xw, L):
		self.xw = xw
		self.L = L
	
	def step(self, dt, B):
		self.tot_samp += B
		D = self.D
		#random samples (z = normalized coord, x = real coord)
		z_samp = np.random.randn(D, B)
		x_samp = self.xw.reshape(D, 1) + np.dot(self.L, z_samp)
		#sample fitness function
		y_samp = self.sample(x_samp)
		
		#differentials in normalized coordinates
		dz = np.average(z_samp*y_samp.reshape(1,B), axis = 1)
		dA = -np.diag(np.ones(D))*np.average(y_samp) + np.average(z_samp.reshape(D, 1, B)*z_samp.reshape(1, D, B)*y_samp.reshape(1, 1,B), axis = 2)
		
		#L_inv = np.linalg.inv(self.L)
		Lamb = np.dot(self.L.T, self.L)
		
		scale = np.trace(self.L)
		dxw =  np.dot(self.L, dz)*1
		
		dL =  np.dot(self.L, dA)*1/D	
		
		#additional factor r is used to help with numerical stability when updating L. 
		#This steps keeps the window from shrinking too quickly.
		L_ = self.L + dt*dL
		r = np.sum(L_**2)**0.5/np.sum(self.L**2)**0.5
		self.L = self.L + r*dt*dL
		
		#expansion of sampling window (experimental)
		#self.L = self.L  + dt*self.g_current*np.diag(np.ones(D))
		#self.L = self.L  + dt*self.g_current*self.L
		
		if(np.average(self.L**2)**0.5 < self.g_current):
			self.L =  self.L*self.g_current/np.average(self.L**2)**0.5
		
		
		#max size of window
		if(np.sqrt(np.average(self.L**2)) > 2):
			self.L =  self.L*2/np.sqrt(np.average(self.L**2))
		
		#update xw
		self.xw = self.xw + r*dt*dxw
	
	
	
	def optimize(self, tot_samp_max = 50000, tr_min = 0):
		self.tot_samp  =0
		tot_samp_rec = []
		xw_rec = []
		L_rec = []
		
		count = 0
		while(self.tot_samp < tot_samp_max and np.abs(np.sum(np.diag(self.L))) > tr_min):
			#batch size chosen as 1/trace(L) here so more accuracy when window shrinks
			B = int(10/np.average(self.L**2)**(0.5*self.kappa))
			#B = 40
			self.g_current = self.g/(count + 1)**self.g_exp
			#B = 4
			self.step(self.dt, B)
			
			#print (for debug)
			if(True and count % 20 == 0):
				print(B)
				print(self.xw)
				print(self.L)
			
			#save info
			tot_samp_rec.append(self.tot_samp)
			xw_rec.append(self.xw)
			L_rec.append(self.L)
			count += 1
		
		return 	tot_samp_rec, xw_rec, L_rec
	
	
	
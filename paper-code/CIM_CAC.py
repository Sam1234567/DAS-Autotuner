#   Author         : <anonymous>
#	Basic coupler and solver for boolean satisfiability (SAT) problem and CAC (chaotic amplitude control) dynamics

import numpy as np
import torch


class CIM:
	
	J = None
	h = None
	norm = 1
	
	dtype = torch.float64
	
	PARAM_NAMES = ["T", "dt", "p_init", "p_end", "beta"]
	#parameters
	T = 1000
	dt = 0.1
	p_init = -1.0
	p_end = 1.0
	beta = 0.25
	
	def __init__(self, pt_device, N, J = None, h = None):
		
		self.pt_device = pt_device
		self.N = N
		self.load_problem(J = J, h = h)
		
	#loads problem from numpy or pytorch tensors J and h (J assumed to be NxN symmetric matrix with zero diagonal)
	def load_problem(self, J = None, h = None):
		if(not J is None):
			self.J = torch.tensor(J, dtype = self.dtype).to(self.pt_device)
			self.N = self.J.shape[0]
		
		if(not h is None):
			self.h = torch.tensor(h, dtype = self.dtype).to(self.pt_device)
		
	
	#loads coupling matrix from numpy array with coordinate representation (i, j, Jij)
	def load_J_coo(self, N, ijJij, h = None):
		self.J = torch.zeros((N,N), dtype = self.dtype)
		
		for (i,j,Jij) in ijJij:
			
			self.J[int(i),int(j)] = Jij
			self.J[int(j),int(i)] = Jij
		
		self.J = self.J.to(self.pt_device)
		
		self.h = None
		
		if(not h is None):
			self.h = torch.tensor(h, dtype = self.dtype).to(self.pt_device)
	
	
	def cal_feedback(self, y):
		if(self.h is None):
			return torch.mm(self.J, y)*self.norm
		return  (torch.mm(self.J, y) + self.h)*norm
	
	def cal_E(self, s):
		if(self.h is None):
			return -0.5*torch.sum(torch.mm(self.J, s)*s, dim=0)
		return -0.5*torch.sum(torch.mm(self.J, s)*s, dim=0) - torch.sum(self.h*s, dim=0)
	
	
	#Init solver with problem and number of trajs
	def init(self, R):
		self.R = R
		
		self.x = torch.randn(self.N, self.R, dtype = self.dtype).to(self.pt_device)
		self.e = torch.ones(self.N, self.R, dtype = self.dtype).to(self.pt_device)
		
		for param_name in self.PARAM_NAMES:
			setattr(self, param_name, torch.tensor(getattr(self, param_name), dtype = self.dtype).to(self.pt_device))
		
		s = torch.sign(self.x)
		self.E = self.cal_E(s)
		self.E_opt = self.E
	
	#step solver with tau = T_current/T_max
	def step(self, tau, update_E = True):
		
		update_flag = (tau <= 1)
		
		#coupling
		y = self.x
		z = self.cal_feedback(y)
		
		#spin amplitude 
		p = self.p_init + (self.p_end - self.p_init)*tau
		x_new = self.x + update_flag*self.dt*( (p-1)*self.x - self.x**3 + self.e*z)
		
		#error amplitude (target amplitude is set to 1)
		self.e = self.e + update_flag*self.dt*self.beta*(1 - self.x**2)
		
		self.x = x_new
		
		#energy calculation
		if(update_E):
			s = torch.sign(self.x)
			self.E = self.cal_E(s)
			flag = self.E < self.E_opt
			self.E_opt = self.E*flag + self.E_opt*(~ flag)
	
	
	#function for annealing schedule (set to linear for now)
	def schedule(self, tau):
		return tau
	
	
	#compute trajectory with target energy. Returns best energy for each trajectory and succes probability. Should be called after init.
	def traj(self, target_E, R_rec = 0):
		
		x_rec = None
		e_rec = None
		E_rec = None
		T_rec = None
		tau_rec = None
		
		if(R_rec > 0):
			T_ = int(np.ceil(self.T))
			x_rec = np.zeros((T_, self.N, R_rec))
			e_rec = np.zeros((T_, self.N, R_rec))
			E_rec = np.zeros((T_, R_rec))
			T_rec = np.zeros(T_)
			tau_rec = np.zeros(T_)
		
		t = 0
		while(t < np.max(np.array(self.T))):
			tau = self.schedule(t/self.T)
			self.step(tau, update_E = True)
			
			if(R_rec > 0):
				x_rec[t ,:, :] = self.x[:, :R_rec]
				e_rec[t ,:, :] = self.e[:, :R_rec]
				E_rec[t, :] = self.E[:R_rec]
				T_rec[t] = t
				tau_rec[t] = tau
			
			t += 1
		
		E_opt_ = self.E_opt.cpu().numpy()
		
		if(target_E is None):
			target_E = np.min(E_opt_)
		
		Ps = np.sum(E_opt_ <= target_E)/self.R
		
		if(R_rec > 0):
			return  Ps, E_opt_, {"x": x_rec, "e": e_rec, "E": E_rec, "T": T_rec, "tau": tau_rec}
		
		
		return Ps, E_opt_
	
	
	
	
	
	
	
	
	
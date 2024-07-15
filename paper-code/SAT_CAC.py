#   Author         : <anonymous>
#	Basic coupler and solver for boolean satisfiability (SAT) problem and CAC (chaotic amplitude control) dynamics

import numpy as np
import torch



class SAT:
	
	#problem info
	M = 1
	K = 1
	#Sparse tensor of shape (N, M, K, K - 1) used for feedback couupling.
	#C
	#Cijkl
	C = None
	IDX = None
	Kjm = None
	Kj = None
	
	dtype = torch.float64
	
	PARAM_NAMES = ["T", "dt", "p_init", "p_end", "beta"]
	#parameters
	T = 1000
	dt = 0.1
	p_init = -1.0
	p_end = 1.0
	beta = 0.25
	
	
	
	#system variables
	x = None
	e = None
	
	
	def __init__(self, pt_device, N, IDX = None, C = None, CNF = None, T = 1000, dt = 0.1, p_init = -1.0, p_end = 1.0, beta = 0.25):
		
		self.pt_device = pt_device
		self.N = N
		
		if(not CNF is None):
			self.load_problem_CNF(N, CNF)
		elif(not IDX is None and not C is None):
			self.load_problem(N, IDX, C)
		
		
		self.T = T
		self.dt = dt
		self.p_init = p_init
		self.p_end = p_end
		self.beta = beta
		
	
	#loads problem from numpy CNF array of shape (M,K)
	def load_problem_CNF(self, N, CNF):
		self.load_problem(N, np.abs(CNF) - 1, np.sign(CNF))
	
	#loads problem using: IDX, numpy array of shape (M,K) containing the indices of literals in each clause and  C, numpy array of shape (M,K) containing literal negation
	def load_problem(self, N, IDX, C):
		self.N = N
		self.M = M = IDX.shape[0]
		self.K = K = IDX.shape[1]
		
		self.C = torch.tensor(C, dtype = self.dtype).to(self.pt_device)
		self.IDX = torch.tensor(IDX, dtype = int).to(self.pt_device)
		
		#sparse matrix for sum over Kij
		IDX_all = np.ravel(IDX)
		ravel_idx = range(len(IDX_all))
		ij = [[int(ij[0]), int(ij[1])] for ij in zip(IDX_all, ravel_idx)]
		
		v = np.ones(len(ij))
		
		ij = np.array(ij, dtype = int).transpose()
		
		C_inv = torch.sparse_coo_tensor(ij, v, (N,K*M), dtype = self.dtype)
		self.C_inv = C_inv.to(self.pt_device)
	
	#this experimental function calculates the SAT feedback (approximately) using exponentials and logarithms.
	def cal_feedback(self, y):
	
		R = y.shape[1]
		
		
		self.Kjm = torch.zeros((self.M, self.K, R), dtype = self.dtype)
	
		
		Kmat = 0.5 - 0.5*self.C.reshape((self.M, self.K, 1))*(y.index_select(0, self.IDX.reshape(self.M*self.K)).reshape(self.M, self.K,-1))
		
		#products over clauses excluding single clause
		for m in range(self.K):
			#print(Kmat[:,:m,:].shape, Kmat[:,m+1:,:].shape)
			self.Kjm[:, m, :] = self.C[:,m].reshape(self.M, 1)*torch.prod(Kmat[:,:m,:], axis = 1)*torch.prod(Kmat[:,m+1:,:], axis = 1)
		
		#sum over clauses
		z = torch.mm(self.C_inv, self.Kjm.reshape(self.M*self.K, -1))
		
		
		return z
		

	
	#this function calculates the SAT energy (exactly) using similar techniques to cal_feedback
	def cal_E(self, s):
		
		self.Kj = torch.prod(0.5 - 0.5*self.C.reshape((self.M, self.K, 1))*(s.index_select(0, self.IDX.reshape(self.M*self.K)).reshape(self.M, self.K,-1)), axis = 1)

		
		return torch.sum(self.Kj, axis = 0)
	
			
		
		
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
	
	
	
	
	
	
	
	
	
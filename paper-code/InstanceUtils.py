import numpy as np
import random

def randSAT(N,M,K):
	IDX = np.zeros((M, K), dtype = int)
	C = np.zeros((M, K))


	for i in range(M):
		for j in range(K):
			IDX[i,j] = np.floor(random.random()*N)
	#         if(i <1):
	#             print(IDX[i,:])
			checked = False
			c = 0
			while(c < j):
				if(IDX[i,j] == IDX[i,c]):
					IDX[i,j] = np.floor(random.random()*N)
					c = -1
				c+= 1
			C[i,j] = np.sign(random.random() - 0.5)
	return C, IDX

def randSK(N):
	J = np.zeros((N,N))
	
	for i in range(N):
		for j in range(i):
			J[i,j] = J[j,i] = np.sign(random.random() - 0.5)
	
	return J

def randGaussianSK(N):
	J = np.zeros((N,N))
	
	for i in range(N):
		for j in range(i):
			J[i,j] = J[j,i] = np.random.randn()
	
	return J

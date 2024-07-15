import numpy as np
#1D gaussian
def f0(x):
	
	
	return np.exp(-10*np.sum(x, axis = 0)**2*1.0 )
	
#1D parabola
def f1(x):
	return np.sum(x, axis = 0)**2*1.0 
	
	
	
#compact support parabolic (2d)
def f2(x):
	return np.maximum(np.average(1 - x**2, axis =0), 00)
	
	
#skewed parabolic sum (2d)
def f3(x):
	return np.maximum(np.average(1 - (1 - 0.9*np.sign(x))*x**2, axis =0), 00)

#skewed parabolic product (2d)
def f4(x):
	return np.prod(np.maximum(1 - (1 - 0.9*np.sign(x))*x**2, 0), axis =0)

#parabolic
def f5(x):
	return np.average(1 - (x-0.1)**2, axis =0)

#pyramidal
def f6(x):
	return np.average(1 - np.abs(x-0.1), axis =0)
	

#conical
def f7(x):
	return 1 - np.sqrt(np.average( (x-0.1)**2, axis =0))


#skewed conical
def f8(x):
	return 1 - np.sqrt(np.average( (1 - 0.9*np.sign(x))*x**2, axis =0))

#skewed quartic
def f9(x):
	return 1 - (np.average( (1 - 0.9*np.sign(x))*x**2, axis =0))**2


#rosenbrock function
def f10(x):
	if(len(x.shape) == 1):
		return -np.sum(100*(x[0:-1]**2 - x[1:])**2 + (x[0:-1]-1)**2)
	
	return -np.sum(100*(x[0:-1, :]**2 - x[1:, :])**2 + (x[0:-1, :]-1)**2, axis = 0)


#gaussian rosenbrock function a = 10
def f11(x):
	beta = 0.5
	if(len(x.shape) == 1):
		return np.exp(-beta*np.sum(10*(x[0:-1]**2 - x[1:])**2 + (x[0:-1]-1)**2))
	
	return np.exp(-beta*np.sum(10*(x[0:-1, :]**2 - x[1:, :])**2 + (x[0:-1, :]-1)**2, axis = 0))


#gaussian rosenbrock function a = 100 beta= 0.5
def f12(x):
	beta = 0.5
	if(len(x.shape) == 1):
		return np.exp(-beta*np.sum(100*(x[0:-1]**2 - x[1:])**2 + (x[0:-1]-1)**2))
	
	return np.exp(-beta*np.sum(100*(x[0:-1, :]**2 - x[1:, :])**2 + (x[0:-1, :]-1)**2, axis = 0))

#gaussian het curvatre
def f13(x):
	beta = np.ones(x.shape[0])
	beta[0] = 100
	if(len(x.shape) == 3):
		return np.exp(-np.sum(beta.reshape(-1,1,1)*x**2, axis = 0))
	if(len(x.shape) == 2):
		return np.exp(-np.sum(beta.reshape(-1,1)*x**2, axis = 0))

	if(len(x.shape) == 1):
		return np.exp(-np.sum(beta.reshape(-1)*x**2, axis = 0))


#quartic gaussian het curvatre
def f14(x):
	beta = np.ones(x.shape[0])
	beta[0] = 100
	if(len(x.shape) == 3):
		return np.exp(-np.sum(beta.reshape(-1,1,1)*x**4, axis = 0))
	if(len(x.shape) == 2):
		return np.exp(-np.sum(beta.reshape(-1,1)*x**4, axis = 0))

	if(len(x.shape) == 1):
		return np.exp(-np.sum(beta.reshape(-1)*x**4, axis = 0))


#gaussian rosenbrock function a = 100 beta = 0.2
def f15(x):
	beta = 0.2
	if(len(x.shape) == 1):
		return np.exp(-beta*np.sum(100*(x[0:-1]**2 - x[1:])**2 + (x[0:-1]-1)**2))
	
	return np.exp(-beta*np.sum(100*(x[0:-1, :]**2 - x[1:, :])**2 + (x[0:-1, :]-1)**2, axis = 0))


#gaussian rosenbrock function a = 100 beta = 0.05
def f16(x):
	beta = 0.05
	if(len(x.shape) == 1):
		return np.exp(-beta*np.sum(100*(x[0:-1]**2 - x[1:])**2 + (x[0:-1]-1)**2))
	
	return np.exp(-beta*np.sum(100*(x[0:-1, :]**2 - x[1:, :])**2 + (x[0:-1, :]-1)**2, axis = 0))





def get_func(idx):
	if(idx == 0):
		return f0
	if(idx == 1):
		return f1
	if(idx == 2):
		return f2
	if(idx == 3):
		return f3
	if(idx == 4):
		return f4
	if(idx == 5):
		return f5
	if(idx == 6):
		return f6
	if(idx == 7):
		return f7
	if(idx == 8):
		return f8
	if(idx == 9):
		return f9
	if(idx == 10):
		return f10
	if(idx == 11):
		return f11
	if(idx == 12):
		return f12
	if(idx == 13):
		return f13
	if(idx == 14):
		return f14
	if(idx == 15):
		return f15
	if(idx == 16):
		return f16
	
	

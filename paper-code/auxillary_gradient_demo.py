import numpy as np

import matplotlib.pyplot as plt




def f(x):
	
	return np.exp(-1.5*(x)**2) + 0.0*np.exp(-0.5*(x-2)**2)
	
def kernel(x):
	return np.exp(-x**2)/np.sqrt(np.pi)
	


x_range_1d = np.arange(-4,4,0.1)


plt.plot(x_range_1d, f(x_range_1d))

plt.show()
plt.close()

x_range = np.arange(-4,4,0.05)
w_range = np.arange(0.05,4,0.05)


M = np.zeros( (len(x_range), len(x_range)))


grid = np.zeros((len(x_range), len(w_range)))
X,W = np.meshgrid(x_range, w_range)

for i in range(len(w_range)):
	w = w_range[i]
	X1, X2 = np.meshgrid(x_range, x_range)
	
	M = 1.0*(kernel( (X1 - X2)/w) )
	M = M/w
	
	grid[:, i] = np.dot(M, f(x_range))


plt.pcolormesh(X, W,grid.T)

plt.show()
plt.close()

f_vals = f(x_range)

def g(w,x):
	return np.sum(f_vals*kernel( (x_range - x) / w))/w


def grad(z):
	v = np.random.randn(2)
	v = v / np.linalg.norm(v)
	v_ = 0.0001*v
	return v*(g(z[0] + v_[0], z[1] + v_[1]) - g(z[0] - v_[0], z[1] - v_[1]))/(2*0.0001)
	



z = np.array([2.0,1.0])

x_list = []
w_list = []

dt = 0.005
for i in range(10000):
	gr = grad(z)
	if(np.linalg.norm(gr) > 0.1):
		gr = 0.1*gr/np.linalg.norm(gr)
	z = z + dt*gr
	z[0] = np.maximum(z[0], 0.05)
	w_list.append(z[0])
	x_list.append(z[1])
	print(z)
	print( (g(z[0] + 0.01 , z[1] + 0.01) - g(z[0] - 0.01, z[1] )) / 0.02)
	print("g", g(z[0], z[1]))

#x_list = (np.array(x_list) - np.min(x_range))*len(x_range)/(np.max(x_range) - np.min(x_range))
#w_list = (np.array(w_list) - np.min(w_range))*len(w_range)/(np.max(w_range) - np.min(w_range))
print(x_list)
print(w_list)
plt.pcolormesh(X, W,grid.T)
#plt.xlim(np.min())
plt.plot(x_list, w_list, color = "red")
plt.xlabel("$x$ (problem variable)")
plt.ylabel("$w$ or $L$ (auxillary window variable)")
plt.show()
plt.close()



import numpy as np

number_neurons:int = 50
input_size = 6
alpha = 0.0
x = np.zeros(number_neurons)
W = np.zeros((number_neurons,number_neurons))
V = np.zeros((number_neurons,input_size))
u = np.zeros(input_size)
W.fill(1.0)
V.fill(1.0)
#V[0][0] = 10.0
#print(V)
u.fill(1.0)

#print(u)
#print(V.dot(u))
print(W.dot(np.tanh(V.dot(u))))
dx_dt = -alpha * x + W.dot(np.tanh(x + V.dot(u)))
#print(dx_dt)

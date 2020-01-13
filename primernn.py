import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import  make_circles

class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
       self.act_f = act_f
       self.b = np.random.rand(1, n_neur) * 2 - 1
       self.w = np.random.rand(n_conn, n_neur) * 2 - 1

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2), lambda Yp , Yr: (Yp - Yr))# Error cuadratico medio

def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):
    # Forward pass
    # -->
    out = [(None, X)]
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].w + neural_net[l].b #Suma ponderada: w1 + w2 + w3 + ...
        a = neural_net[l].act_f[0](z)
        out.append((z, a))
    # Backpropagation
    # Gradiente descendiente
    # <--
    if train:
        # Backpropagation
        deltas = []
        for l in reversed(range(0, len(neural_net))):
            z = out[l + 1][0]
            a = out[l + 1][1]

            if l == len(neural_net) - 1: # Para la ultima capa el back se calcula distinto
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[1].act_f[1](a)) 
            else:
                deltas.insert(0, deltas[0] @ _w.T * neural_net[l].act_f[1](a))
            
            _w = neural_net[l].w

            # Gradiente descendiente (Optimizar o corregir pesos)
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr 
            neural_net[l].w = neural_net[l].w - out[l][1].T @ deltas[0] * lr

    return out[-1][1]

def create_nn(topology, act_f):
    nn = [] # Vector de capas
    for l, layer in enumerate(topology[:-1]):
       nn.append(neural_layer(topology[l], topology[l+1], act_f))
    return nn

# Funciones de activacion
sigm = (lambda x: 1 / (1 + np.e ** (-x)), lambda x: x * (1 - x))
relu = lambda x: np.maximun(0, x)

# Dataset
n = 500 # Population
p = 2   # Characteristics
from sklearn.datasets import make_blobs
X, Y = make_blobs(n_samples=n,cluster_std=0.5, n_features=2, centers=[(-1, 1), (1,1)])
Y = Y[:, np.newaxis]
#show info
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], color ='salmon')
plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], color = 'skyblue')
plt.axis('equal')
plt.show()

topology = [p, 4, 8, 4, 1] # [in, hidden, out]

# Creamos la red y la probamos!
import time
from tqdm import tqdm
neural_net = create_nn(topology, sigm)

loss = [] # Vector para guardar los errores

for i in tqdm(range(5000), desc='Training nn'):
    # Entrenamos la red!
    pY = train(neural_net, X, Y, l2_cost)
    if i % 25 == 0:
        loss.append(l2_cost[0](pY, Y))
        res = 50
        _x0 = np.linspace(-10, 10, res)
        _x1 = np.linspace(-10, 10, res)

        _Y = np.zeros((res, res))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0, i1] = train(neural_net, np.array([[x0, x1]]), Y, l2_cost, train = False)[0][0]

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(17,10))
ax0.pcolormesh(_x0, _x1, _Y, cmap='coolwarm')

ax0.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], color ='salmon')
ax0.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], color = 'skyblue') 
ax1.plot(range(len(loss)), loss)
ax0.axis('equal')
ax1.axis('equal')
plt.show()

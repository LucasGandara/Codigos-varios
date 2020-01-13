import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

#Funciona a optimizar
func = lambda th: np.sin(1/2 * th[0] ** 2 - 1/4 * th[1] ** 2 + 3) * np.cos(2 * th[0] + 1 - np.e ** th[1])

res = 100
_X = np.linspace(-2, 2, res)
_Y = np.linspace(-2, 2, res)

_Z = np.zeros((res, res))

for ix, x in enumerate(_X):
    for iy, y in enumerate(_Y):
        _Z[iy, ix] = func([x, y])

# Vemos el mapa 2D
plt.contourf(_X, _Y, _Z, 100)
plt.colorbar()

Theta = np.random.rand(2) * 4 - 2 # Valor dentro del plano

plt.plot(Theta[0], Theta[1], 'o', color='white')

_T = np.copy(Theta)
h = 0.001
grad = np.zeros(2)
lr = 0.001
for i in range(10000):
    for it, th in enumerate(Theta):
        _T = np.copy(Theta)
        _T[it] += h
        deriv = (func(_T) - func(Theta)) / h # Formula de la derivada
        grad[it] = deriv

    Theta -= lr * grad
    if i % 100 == 0:
        plt.plot(Theta[0], Theta[1] ,'.', color='red')
plt.plot(Theta[0], Theta[1], 'o', color='green')
plt.show()

""" Si quieres ver cambios en el comportamiento significativos, juega
    con la variable lr...."""

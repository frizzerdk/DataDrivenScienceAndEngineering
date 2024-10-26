import numpy as np
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt
x = np.arange(-5,5.01,0.1)
y = np.arange(-6,6.01,0.1)
t = np.arange(0,10*np.pi+0.1,0.1)

X,Y,T = np.meshgrid(x,y,t)

A = (np.exp(-(X**2+ 0.5*Y**2)) * np.cos(2*T) +
    (np.divide(np.ones_like(X),np.cosh(X)) * np.tanh(X) * np.exp(-0.2*Y**2)) * np.sin(T))

print(A.shape)

out_A = parafac(A,2)

A1, A2, A3 = out_A.factors

print(A1.shape, A2.shape, A3.shape)

plt.figure(figsize=(12,6))

plt.subplot(3,1,1)
plt.plot(A1)

plt.subplot(3,1,2)
plt.plot(A2)

plt.subplot(3,1,3)
plt.plot(A3)

plt.show()

n_rows, n_cols = 10, 10
n_plots = n_rows * n_cols
t_max = 10*np.pi
t_indices = np.linspace(0, len(t)-1, n_plots, dtype=int)

plt.figure(figsize=(15, 15))
for i, t_idx in enumerate(t_indices):
    plt.subplot(n_rows, n_cols, i+1)
    plt.imshow(A[:,:,t_idx])
    plt.title(f't = {t[t_idx]:.2f}')
    plt.axis('off')

plt.tight_layout()
plt.show()

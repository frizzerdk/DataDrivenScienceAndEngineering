import numpy as np
import matplotlib.pyplot as plt

# Create a single figure for all plots
plt.figure(figsize=(12, 8))
# make ax1 and ax2
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

# Generate a point cloud
xCenter = np.array([2,1])
singular_values = np.array([2,0.5])

theta = np.pi/3
Rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])

nPoints = 100000

X = (Rotation_matrix @ np.diag(singular_values) @ np.random.randn(2, nPoints) 
    + np.diag(xCenter) @ np.ones((2, nPoints)))

ax1.plot(X[0,:], X[1,:], '.', color='k', markersize=1)
Xavg = np.mean(X, axis=1)
B = X - np.tile(Xavg, (nPoints, 1)).T

U, S, VT = np.linalg.svd(B/np.sqrt(nPoints-1), full_matrices=0)

theta = 2*np.pi*np.arange(0, 1, 0.01)
Xstd = U @ np.diag(S) @ np.array([np.cos(theta), np.sin(theta)]) 

ax2.plot(Xavg[0], Xavg[1], 'x', color='k', markersize=10)
ax2.plot(X[0,:], X[1,:], '.', color='k', markersize=1, alpha=0.05)
ax2.plot(Xavg[0] + Xstd[0,:], Xavg[1] + Xstd[1,:], '-',color='r', linewidth=2) # 1 standard deviation
ax2.plot(Xavg[0] + 2*Xstd[0,:], Xavg[1] + 2*Xstd[1,:], '-', color='g', linewidth=2) # 2 standard deviations
ax2.plot(Xavg[0] + 3*Xstd[0,:], Xavg[1] + 3*Xstd[1,:], '-', color='b', linewidth=2) # 3 standard deviations

plt.show()

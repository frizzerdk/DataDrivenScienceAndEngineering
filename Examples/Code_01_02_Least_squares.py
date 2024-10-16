import numpy as np
import matplotlib.pyplot as plt
# Define the data points

samples = 5
noise_apmlitude = 10
a_range = 10

x = 3
a = np.arange(-a_range, a_range, (2*a_range)/samples).reshape(-1, 1)

# Create a single figure for all plots
plt.figure(figsize=(12, 8))

# Plot true data
plt.plot(a, x*a, color='k', linewidth=2, label='True Data')

# Generate and plot 5 examples of noisy data and estimates
colors = ['r', 'g', 'b', 'm', 'c']
for i in range(5):
    # Generate noisy data
    b = a*x + np.random.randn(*a.shape)*noise_apmlitude
    
    # Plot noisy data
    plt.plot(a, b, 'x', color=colors[i], markersize=5, label=f'Noisy Data {i+1}')
    
    # Compute SVD and estimate
    U, S, VT = np.linalg.svd(a, full_matrices=False)
    xtilde = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
    
    # Plot estimated data
    plt.plot(a, xtilde*a, color=colors[i], linewidth=1, label=f'Estimated Data {i+1}')

plt.legend()
plt.xlabel('a')
plt.ylabel('b')
plt.title('Least Squares Estimation: 5 Examples')
plt.grid(True)
plt.show()
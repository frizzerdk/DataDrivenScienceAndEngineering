import matplotlib
matplotlib.use('TkAgg')  # Add this line before importing pyplot

from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os

filedir = os.path.dirname(os.path.abspath(__file__))
A = imread(filedir + '/Inputs/Fuji.jpg')
X = np.mean(A, -1)

plt.figure(figsize=(10, 8),)
img = plt.imshow(X, cmap='gray')
plt.axis('off')
plt.title('Grayscale Image of Mount Fuji')
plt.tight_layout()

# Create a figure with subplots for images and error plot
fig, axs = plt.subplots(4, 3, figsize=(20, 12))
fig.suptitle('Image Compression with SVD', fontsize=16)

# Flatten the axes for easier indexing
axs = axs.flatten()

errors = []

# Compress the image with different numbers of singular values
r_values = (np.logspace(0, 2, num=10))  # Log scale from 1% to 100%
for i, r in enumerate(r_values):
    print(f'Compressing with {r:.2f}% of the singular values')
    U, S, V = np.linalg.svd(X, full_matrices=False)
    r_actual = int(r * len(S) / 100)  # Convert percentage to actual number of singular values
    Xapprox = U[:, :r_actual] @ np.diag(S[:r_actual]) @ V[:r_actual, :]
    
    # Plot the compressed image
    axs[i].imshow(Xapprox, cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(f'r={r:.2f}%')
    
    # Calculate and store the error
    error = np.linalg.norm(X - Xapprox, 'fro') / np.linalg.norm(X, 'fro')
    errors.append(error)
    print(f'Error: {error}')

# Plot the error
ax_error = axs[-2]  # Use the second to last subplot for the error plot with log scale
ax_error.semilogx(r_values, errors, 'bo-')
ax_error.set_xlabel('% Singular Values')
ax_error.set_ylabel('Error')
ax_error.set_title('Compression Error (Log Scale)')

# Add linear axis plot
ax_linear = axs[-1]  # Use the last subplot for the error plot with linear scale
ax_linear.plot(r_values, errors, 'ro-')
ax_linear.set_xlabel('% Singular Values')
ax_linear.set_ylabel('Error')
ax_linear.set_title('Compression Error (Linear Scale)')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def randomizedSVD(X, r, q, p):
    ny = X.shape[1]
    P = np.random.randn(ny, r+p)
    Z = X @ P
    for k in range(q):
        Z =  X @ ( X.T @ Z)
    Q, R = np.linalg.qr(Z , mode='reduced')

    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices=False)
    U = Q @ UY
    return U, S, VT   

if __name__ == "__main__":
    # load image 
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    path = "Inputs/Fuji.jpg"
    image = plt.imread(path)
    X = np.mean(image, axis=2)
    fig = plt.figure(figsize=(30,20))
    plt.subplot(1,2,1)
    plt.imshow(X, cmap='gray')
    print(X.shape)

    r_values = [ 400]
    p_values = [10]
    q_values = [0, 1, 2, 5, 10, 20]
    fig, axes = plt.subplots(len(r_values), len(q_values), figsize=(20, 10))

    for i, r in enumerate(r_values):
        for j, q in enumerate(q_values):
            U, S, VT = randomizedSVD(X, r, q, 10)
            reconstructed_image = U @ np.diag(S) @ VT
            reconstructed_image = reconstructed_image.reshape(image.shape[0], image.shape[1])
            ax = axes[i, j]
            ax.imshow(reconstructed_image, cmap='gray')
            ax.set_title(f'r={r}, q={q}')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

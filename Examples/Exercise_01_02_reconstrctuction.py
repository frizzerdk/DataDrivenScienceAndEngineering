import numpy as np
import matplotlib.pyplot as plt
import os
import diskcache as dc
import time
import functools
import hashlib

cache = dc.Cache("cache", size_limit=1e9)
tic = time.time()
filedir = os.path.dirname(os.path.abspath(__file__))
A = plt.imread(filedir + "/Inputs/Fuji.jpg")
X = np.mean(A, axis=-1)

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.imshow(X, cmap='gray')
plt.axis('off')
plt.title('Grayscale Image of Mount Fuji')
plt.tight_layout()

def hash_args_cache(cache):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, force_new=False, **kwargs):
            # Create hasher
            hasher = hashlib.sha256()
            func_name = func.__name__
            # Hash positional arguments
            for arg in args:
                if isinstance(arg, np.ndarray):
                    hasher.update(arg.tobytes())
                else:
                    hasher.update(str(arg).encode())
            
            # Hash keyword arguments (sorted to ensure consistent ordering)
            for key, value in sorted(kwargs.items()):
                hasher.update(key.encode())
                if isinstance(value, np.ndarray):
                    hasher.update(value.tobytes())
                else:
                    hasher.update(str(value).encode())
            
            hasher.update(func_name.encode())
            # Get hash key
            key = hasher.hexdigest()
            
            # Try to get from cache, unless force_new is True
            result = None if force_new else cache.get(key, default=None)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result)
            
            return result
        return wrapper
    return decorator

@hash_args_cache(cache)
def get_svd(X):
    return np.linalg.svd(X, full_matrices=False)

@hash_args_cache(cache)
def relative_reconstruction_error(X, U, S, VT, rank: int):
    Xapprox = U[:, :rank] @ np.diag(S[:rank]) @ VT[:rank, :]
    abs_error = np.linalg.norm(X - Xapprox, 'fro')
    rel_error = abs_error / np.linalg.norm(X, 'fro')
    return rel_error, abs_error

@hash_args_cache(cache)
def get_errors_for_ranks(X, U, S, VT, r_values):
    rel_errors = []
    print(f'Computing {len(r_values)} relative reconstruction errors\n')
    for r in r_values:
        print(f'Computing relative reconstruction error for rank {r}', end='\r')
        rel_error, abs_error = relative_reconstruction_error(X, U, S, VT, r)
        rel_errors.append(rel_error)
    print('\n')
    return rel_errors

U, S, VT = get_svd(X)

max_rank = np.min(X.shape)
print(f'Max rank: {max_rank}')
r_values = np.arange(1, max_rank+1)

rel_errors = get_errors_for_ranks(X, U, S, VT, r_values)
rel_vars = np.square(np.array(rel_errors))

print(f'Computing {len(r_values)} relative reconstruction errors\n')
# for i, r in enumerate(r_values.astype(int)):
#     print(f'Computing relative reconstruction error for rank {r}', end='\r')
#     rel_error, abs_error = relative_reconstruction_error(X, U, S, VT, r,force_new=False)
#     rel_errors.append(rel_error)

singular_values = S
cum_singular_values = np.cumsum(singular_values)
rel_singular_values = cum_singular_values / cum_singular_values[-1]
singular_values_99 = np.where(rel_singular_values >= 0.99)[0][0]
print(f'99% of the singular values is captured by rank {singular_values_99}')
rel_singular_values_squared = np.square(rel_singular_values)
rel_singular_values_squared_99 = np.where(rel_singular_values_squared >= 0.99)[0][0]
print(f'99% of the singular values squared is captured by rank {rel_singular_values_squared_99} ')
# find 99% of the variance
vars = 1-np.array(rel_vars)
rank_99 = np.where(vars >= 0.99)[0][0]
print(f'99% of the variance is captured by rank {rank_99} ')

print(f'Time taken: {time.time() - tic:.2f} seconds')
plt.subplot(2,2,2)
plt.plot(r_values, 1-np.array(rel_errors))
plt.plot(r_values, 1-np.array(rel_vars), '-', color='black')
plt.plot(r_values, rel_singular_values, '-', color='blue')
plt.plot(r_values, rel_singular_values_squared, '-', color='green')
# 99% of the variance x and y lines
plt.axvline(rank_99, color='black', linestyle='--')
plt.axhline(0.99, color='black', linestyle='--')
plt.axvline(singular_values_99, color='black', linestyle='--')
plt.axvline(rel_singular_values_squared_99, color='black', linestyle='--')
plt.legend(['Relative Error', 'Relative Variance', 'Singular Values', '99% of Variance', '99% of Singular Values Squared'])

plt.xlabel('Rank')
plt.ylabel('Relative Error')
plt.title('Reconstruction Error vs Rank')
plt.tight_layout()

r1_reconstructed = U[:, :1] @ np.diag(S[:1]) @ VT[:1, :]
plt.subplot(2,2,3)
plt.imshow(r1_reconstructed, cmap='gray')
plt.axis('off')
plt.title('Reconstructed Image at Rank 1')
plt.tight_layout()

# for i, r in enumerate(r_values.astype(int)):    
#     subplot_index = n_examples + i + 1
#     cols = int(n_examples/2)
#     plt.subplot(4,cols,subplot_index)
#     plt.imshow(Xapproxs[i], cmap='gray')
#     plt.axis('off')
#     plt.title('Reconstructed Image at Rank ' + str(r))

print(f'Time taken: {time.time() - tic:.2f} seconds')
#plt.tight_layout()
plt.show()
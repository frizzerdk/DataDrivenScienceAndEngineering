import kagglehub
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Download latest version
path = kagglehub.dataset_download("tbourton/extyalebcroppedpng")

'''
About this directory
The cropped dataset only contains the single P00 pose.

Data format is like yaleBxx_P00A(+/-)aaaE(+/-)ee

xx = Subject ID
(+/-)aaa = Azimuth angle
(+/-)ee = Elevation angle
For example the file yaleB38_P00A+035E+65.png is of subject 38, in pose 00, with light source at (+035, +65) degrees (azimuth, elevation) w.r.t the camera.
'''
print("Path to dataset files:", path)

# load faces as a numpy array
dict_faces = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
image_folder = path + "/CroppedYalePNG/"
for file in os.listdir(image_folder):
    if file.endswith(".png") and "ambient" not in file.lower():
        # Extract metadata from filename
        parts = file.split('_')
        subject_id = parts[0][5:]  # Remove 'yaleB' prefix
        pose, angles = parts[1].split('A')
        
        azimuth, elevation = map(lambda x: int(x.replace('+', ' ').replace('-', ' -')), angles.replace('.png', '').split('E'))
        
        # Load image
        image_array = plt.imread(os.path.join(image_folder, file))
        
        # Store image in nested dictionary
        dict_faces[subject_id][pose][azimuth][elevation] = image_array

#default lighting
default_azimuth = 0
default_elevation = 0

# make matrix of all faces 
all_faces = []
all_faces_column = []
n_images = []
for i, subject_id in enumerate(dict_faces):
    n_images.append(0)
    for azimuth in dict_faces[subject_id]['P00']:
        for elevation in dict_faces[subject_id]['P00'][azimuth]:
            all_faces.append(dict_faces[subject_id]['P00'][azimuth][elevation])
            all_faces_column.append(dict_faces[subject_id]['P00'][azimuth][elevation].flatten())
            n_images[i] += 1

print("n_images shape:", np.array(n_images).shape)
print("all_faces shape:", np.array(all_faces).shape)    

training_subjects = np.array(all_faces[:sum(n_images[:36])])
print("training_subjects shape:", training_subjects.shape)
plt.subplot(2, 2, 1)
plt.imshow(training_subjects[0].reshape(192,168), cmap='gray')
plt.axis('off')
plt.title("Training subject 0")

# change dimension so it's MxN by faces from faces by m by n
training_subject_column = np.array(all_faces_column[:sum(n_images[:36])]).T
print("training_subject_column shape:", training_subject_column.shape)


mean_face_column = np.mean(training_subject_column, axis=1)
print("mean_face_column shape:", mean_face_column.shape)

X_mean = np.tile(mean_face_column,(training_subject_column.shape[1],1))
print("X_mean shape:", X_mean.shape)
X = training_subject_column - X_mean.T
print("X shape:", X.shape)

faces =  np.array(all_faces_column).T
print("faces shape:", faces.shape)
testFace = faces[:,np.sum(n_images[:36])+1]
subject_id = "38"  
pose = "P00"
azimuth = 0
elevation = 0
testFace = dict_faces[subject_id][pose][azimuth][elevation]
testFace = testFace.flatten()
print("testFace shape:", testFace.shape)
testFaceMeanSubtracted = testFace-mean_face_column
print("testFaceMeanSubtracted shape:", testFaceMeanSubtracted.shape)
# compute SVD if not already computed
if not os.path.exists("U.npy"):
    U, S, VT = np.linalg.svd(X, full_matrices=0)
    np.save("U.npy", U)
    np.save("S.npy", S)
    np.save("VT.npy", VT)
else:
    U = np.load("U.npy")
    S = np.load("S.npy")
    VT = np.load("VT.npy")

print("U shape:", U.shape)
print("S shape:", S.shape)
print("VT shape:", VT.shape)


r_list = [25, 50, 100, 200, 400, 800, 1600]

plt.figure(figsize=(15, 3))
plt.subplot(1, len(r_list)+1, 1)
plt.imshow(testFace.reshape(192,168), cmap='gray')
plt.axis('off')
plt.title("Test face")
for i, r in enumerate(r_list):
    reconstructed_face = mean_face_column + U[:,:r] @ U[:,:r].T @ testFaceMeanSubtracted
    plt.subplot(1, len(r_list)+1, i+2)
    plt.imshow(reconstructed_face.reshape(192,168), cmap='gray')
    plt.axis('off')
    plt.title("Reconstructed face with r = " + str(r))

plt.show()
print("done")

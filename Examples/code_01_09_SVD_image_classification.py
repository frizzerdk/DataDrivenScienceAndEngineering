import kagglehub
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.cm as cm

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


# Foreach person plt the 5th and 6th pca mode
face_array = np.array(all_faces_column).T
person_filter = [3, 36]
Persons = []
for i, subject_id in enumerate(person_filter):
    P = face_array[:,np.sum(n_images[:subject_id]):np.sum(n_images[:subject_id+1])]
    P = P - np.tile(mean_face_column,(P.shape[1],1)).T
    Persons.append(P)

# find best separation of persons
def loss_function(pca_modes,same_person_weight=1,different_person_weight=1):
    distances_to_different_person = []
    distances_to_same_person = []
    PCA_cords = []
    
    for i, person in enumerate(Persons):
        pca_cords = U[:,pca_modes].T @ person
        PCA_cords.append(pca_cords)
    
    #normalize PCA_cords coordinates, first find the max and min of each coordinate accross all persons
    all_pca_cords = np.concatenate(PCA_cords, axis=1)
    max_pca_cords = np.max(all_pca_cords, axis=1)
    min_pca_cords = np.min(all_pca_cords, axis=1)
    range_pca_cords = max_pca_cords - min_pca_cords
    normalized_pca_cords = (PCA_cords - min_pca_cords.reshape(-1,1)) / range_pca_cords.reshape(-1,1)

    for i, PCA_cord in enumerate(normalized_pca_cords):
        for j, other_PCA_cord in enumerate(normalized_pca_cords):
            if i != j:
                for k in range(PCA_cord.shape[1]):
                    distances_to_different_person.append(np.linalg.norm(PCA_cord[:,k] - other_PCA_cord[:,k]))
    for i, PCA_cord in enumerate(normalized_pca_cords):
        for k in range(PCA_cord.shape[1]):
            distances_to_same_person.append(np.linalg.norm(PCA_cord[:,k] - normalized_pca_cords[i][:,k]))
    return np.mean(distances_to_different_person)*different_person_weight, np.mean(distances_to_same_person)*same_person_weight

# test random pca modes between 5 and 30
total_loss = []
pca_modes = []
pca_cords = []
max_pca_index = 2000
min_pca_index = 0
tries = 10000
for _ in range(tries):  # Increased from 9 to 100
    random_pca_modes = np.random.choice(range(min_pca_index, max_pca_index), 3, replace=False)
    loss_different_person, loss_same_person = loss_function(random_pca_modes)
    total_loss.append(-loss_different_person + loss_same_person)
    pca_modes.append(random_pca_modes)
    pca_cords.append([U[:, random_pca_modes].T @ person for person in Persons])
print(f"Total loss: {np.mean(total_loss)}")

sorted_indices = np.argsort(total_loss)
best_indices = sorted_indices[:6]  # 6 best (lowest loss)
worst_indices = sorted_indices[-6:]  # 6 worst (highest loss)

colors = cm.rainbow(np.linspace(0, 1, len(Persons)))

fig = plt.figure(figsize=(20, 24))
fig.suptitle("PCA Mode Combinations for Person Separation", fontsize=16, y=0.98)

def plot_pca_combination(index, plot_index, is_best):
    ax = fig.add_subplot(3, 4, plot_index, projection='3d')
    
    for i, person_pca_cords in enumerate(pca_cords[index]):
        plot_color = colors[i]
        ax.scatter(person_pca_cords[0, :], person_pca_cords[1, :], person_pca_cords[2, :], c=[plot_color], marker='o', label=f'Person {i}')
    
    ax.set_xlabel(f'PCA {pca_modes[index][0]}')
    ax.set_ylabel(f'PCA {pca_modes[index][1]}')
    ax.set_zlabel(f'PCA {pca_modes[index][2]}')
    ax.set_title(f'{"Best" if is_best else "Worst"} #{plot_index if is_best else plot_index-6}\nLoss: {total_loss[index]:.4f}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize='x-small')

# Plot best combinations
for i, index in enumerate(best_indices):
    plot_pca_combination(index, i + 1, True)

# Plot worst combinations
for i, index in enumerate(worst_indices):
    plot_pca_combination(index, i + 7, False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(hspace=0.4, wspace=0.3)
# plot the people used
fig = plt.figure(figsize=(10, 10))
for i, person in enumerate(Persons):
    plt.subplot(2, 2, i+1)
    plt.imshow(person[:,0].reshape(192,168), cmap='gray')
    plt.axis('off')
    plt.title(f"Person {person_filter[i]}")
plt.show()

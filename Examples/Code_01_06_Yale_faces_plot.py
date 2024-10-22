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
        
        azimuth, elevation = map(lambda x: int(x.replace('+', '').replace('-', '')), angles.replace('.png', '').split('E'))
        
        # Load image
        image_array = plt.imread(os.path.join(image_folder, file))
        
        # Store image in nested dictionary
        dict_faces[subject_id][pose][azimuth][elevation] = image_array

# Function to recursively list keys in nested dictionary
def list_keys_recursive(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key))
        if isinstance(value, dict):
            list_keys_recursive(value, indent + 1)

# List keys in dict_faces recursively
print("Structure of dict_faces:")
list_keys_recursive(dict_faces)
# Plotting 36 subjects
n, m = 192, 168
allPersons = np.zeros((n*6, m*6))
subject_ids = list(dict_faces.keys())
for i in range(6):
    for j in range(6):
        plot_subject = subject_ids[i*6 + j]
        allPersons[i*n:(i+1)*n, j*m:(j+1)*m] = dict_faces[plot_subject]['P00'][00][00]

plt.imshow(allPersons, cmap='gray')
plt.axis('off')
plt.show()

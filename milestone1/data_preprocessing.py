# Go through every folder and save the files into input-output numpy arrays

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
from sklearn.model_selection import train_test_split

# Specify the path to the main folder containing 'test' and 'train' subfolders.
main_folder = 'ACDC/database'

# Initialize empty lists to store the data
input = []
output = []

#Function to load nifti image
def load_images(file_path):
  img = nib.load(file_path).get_fdata()
  return img

# Function to check if a file should be ignored
def should_ignore(file_name):
    return file_name == 'MANDATORY_CITATION.md'

# Iterate through 'test' and 'train' folders
for folder_name in ['testing', 'training']:
    if not should_ignore(folder_name):
      folder_path = os.path.join(main_folder, folder_name)
      patient_folders = os.listdir(folder_path)

      # Iterate through patient folders
      for patient_folder in patient_folders:
        if not should_ignore(patient_folder):
          patient_path = os.path.join(folder_path, patient_folder)
          files = os.listdir(patient_path)

          # Iterate through files in each patient folder
          for file_name in files:
            file_path = os.path.join(patient_path, file_name)

            if file_name.endswith('.nii.gz') & ('4d' not in file_name):
              #Separate the gt and non-gt files
              if 'gt' in file_name:
                gt_img = load_images(file_path)
                output.append([folder_name, gt_img])
              else:
                img = load_images(file_path)
                input.append([folder_name, img])

input_array = np.array(input)
output_array = np.array(output)

print(f"input shape: {input_array.shape}")
print(f"output shape: {output_array.shape}")
print(f"input folders: {input_array[:10,0]}")
print(f"input data: {input_array[:3,1]}")
print(f"output data: {output_array[:3,1]}")

# Find the largest shape among all the arrays
largest_input_shape = max([arr.shape for arr in input_array[:,1]])
largest_output_shape = max([arr.shape for arr in output_array[:,1]])

# Iterate through the arrays and pad them
def padding(array, largest_shape):
  padded_arrays = []
  for i, arr in enumerate(array[:,1]):
      # Calculate the padding for the first two dimensions only, keep the last dimension the same
      pad_width = [(0, max_dim - curr_dim) if idx < 2 else (0, 0) for idx, (max_dim, curr_dim) in enumerate(zip(largest_shape, arr.shape))]

      # Pad the array with zeros
      padded_arr = np.pad(arr, pad_width, mode='constant')

      # Append the padded array to the new list
      padded_arrays.append([array[i,0], padded_arr])

  # Convert the list of padded arrays back to a NumPy array
  padded_arrays = np.array(padded_arrays)

  return padded_arrays

# Complete the padding
padded_inputs = padding(input_array, largest_input_shape)

# THE NOTEBOOK DIES HERE
padded_outputs = padding(output_array, largest_output_shape)

#print(padded_inputs.shape)
#print(padded_outputs.shape)

# Look at the shapes of each image

for i, data in enumerate(padded_inputs[:,1]):
    print(f"Image {i + 1} shape: {data.shape}")

#Separate the inputs and outputs into train-test-val arrays

print(f"Input array shape: {padded_inputs.shape}")

input_train = padded_inputs[np.where(padded_inputs[:, 0] == 'training')]
output_train = padded_outputs[np.where(padded_outputs[:, 0] == 'training')]

print(f"Input train shape: {input_train.shape}")
print(f"Output train shape: {output_train.shape}")

input_test = padded_inputs[np.where(padded_inputs[:, 0] == 'testing')]
output_test = padded_outputs[np.where(padded_outputs[:, 0] == 'testing')]
input_val, input_test, output_val, output_test = train_test_split(input_test, output_test, train_size = 0.6, shuffle = False)

print(f"Input validation shape: {input_val.shape}")
print(f"Output validation shape: {output_val.shape}")
print(f"Input test shape: {input_test.shape}")
print(f"Output test shape: {output_test.shape}")
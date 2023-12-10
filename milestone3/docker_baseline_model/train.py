# Go through every folder and save the files into input-output numpy arrays

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F
from torchmetrics import Dice
from segmentation_models_pytorch.losses import DiceLoss

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

input_train = input_array[np.where(input_array[:, 0] == 'training')]
output_train = output_array[np.where(output_array[:, 0] == 'training')]

input_test = input_array[np.where(input_array[:, 0] == 'testing')]
output_test = output_array[np.where(output_array[:, 0] == 'testing')]
input_val, input_test, output_val, output_test = train_test_split(input_test, output_test, train_size = 0.6, shuffle = False)

# NEW PADDING

# Find the largest shape among all the arrays
largest_input_shape = np.max([arr.shape for _, arr in input_array], axis=0)
largest_output_shape = np.max([arr.shape for _, arr in output_array], axis=0)

# Function to pad and process a batch of arrays
def process_batch(array, largest_shape, batch_size):
    padded_arrays = []
    for i in range(0, len(array), batch_size):
        batch = array[i:i + batch_size]
        batch_padded = []
        for name, arr in batch:
            max_shape = np.maximum(largest_shape, arr.shape)
            pad_width = [(0, max_dim - curr_dim) for max_dim, curr_dim in zip(max_shape, arr.shape)]
            padded_arr = np.pad(arr, pad_width, mode='constant')
            batch_padded.append((name, padded_arr))
        padded_arrays.extend(batch_padded)
    return padded_arrays
    
    
# Assuming input_array has text in the first column and 3D images in the second column
# Function to slice 3D images along the third axis and create 2D images
def slice_3d_images(input_array):
    sliced_images = []
    for text, image_3d in input_array:
        for i in range(image_3d.shape[2]):
            image_2d = image_3d[:, :, i].astype('float16')
            sliced_images.append(image_2d)
    return np.array(sliced_images)
    
    
# Slice 3D images to get 2D images
sliced_input_train = slice_3d_images(input_train)
sliced_input_val = slice_3d_images(input_val)
sliced_input_test = slice_3d_images(input_test)

sliced_output_train = slice_3d_images(output_train)
sliced_output_val = slice_3d_images(output_val)
sliced_output_test = slice_3d_images(output_test)


sliced_input_train = sliced_input_train[:100]
sliced_output_train = sliced_output_train[:100]
sliced_input_val = sliced_input_val[:40]
sliced_output_val = sliced_output_val[:40]
sliced_input_test = sliced_input_test[:40]
sliced_output_test = sliced_output_test[:40]

print(sliced_input_train.shape)
print(sliced_input_val.shape)
print(sliced_input_test.shape)
print(sliced_output_train.shape)
print(sliced_output_val.shape)
print(sliced_output_test.shape)

sliced_inputs_all = np.concatenate((sliced_input_train, sliced_input_val, sliced_input_test))
print(f"sliced_inputs_all shape: {sliced_inputs_all.shape}")

max_width = max(image.shape[1] for image in sliced_inputs_all)
max_height = max(image.shape[0] for image in sliced_inputs_all)

print(f"Max width: {max_width}, Max height: {max_height}")

# Round up the maximum height to be divisible by 32
padded_height = ((max_height - 1) // 32 + 1) * 32
print(f"Padded height (to be divisible by 32): {padded_height}")

# Pad each 2D image to match the maximum width and height
padded_inputs = []
for image in sliced_input_train:
    pad_width = ((0, padded_height - image.shape[0]), (0, max_width - image.shape[1]))
    padded_image = np.pad(image, pad_width, mode='constant')
    padded_inputs.append(padded_image)

# Convert the list of padded images to a NumPy array
padded_input_train = np.array(padded_inputs)

# Now padded_images_array contains all the 2D images padded to the desired dimensions
print(padded_input_train.shape)  # The shape will be (total_2d_images, padded_height, padded_width)

# Pad each 2D image to match the maximum width and height
padded_inputs = []
for image in sliced_output_train:
    pad_width = ((0, padded_height - image.shape[0]), (0, max_width - image.shape[1]))
    padded_image = np.pad(image, pad_width, mode='constant')
    padded_inputs.append(padded_image)

# Convert the list of padded images to a NumPy array
padded_output_train = np.array(padded_inputs)

# Now padded_images_array contains all the 2D images padded to the desired dimensions
print(padded_output_train.shape)

# Pad each 2D image to match the maximum width and height
padded_inputs = []
for image in sliced_input_val:
    pad_width = ((0, padded_height - image.shape[0]), (0, max_width - image.shape[1]))
    padded_image = np.pad(image, pad_width, mode='constant')
    padded_inputs.append(padded_image)

# Convert the list of padded images to a NumPy array
padded_input_val = np.array(padded_inputs)

# Now padded_images_array contains all the 2D images padded to the desired dimensions
print(padded_input_val.shape)

# Pad each 2D image to match the maximum width and height
padded_inputs = []
for image in sliced_output_val:
    pad_width = ((0, padded_height - image.shape[0]), (0, max_width - image.shape[1]))
    padded_image = np.pad(image, pad_width, mode='constant')
    padded_inputs.append(padded_image)

# Convert the list of padded images to a NumPy array
padded_output_val = np.array(padded_inputs)

# Now padded_images_array contains all the 2D images padded to the desired dimensions
print(padded_output_val.shape)

# Pad each 2D image to match the maximum width and height
padded_inputs = []
for image in sliced_input_test:
    pad_width = ((0, padded_height - image.shape[0]), (0, max_width - image.shape[1]))
    padded_image = np.pad(image, pad_width, mode='constant')
    padded_inputs.append(padded_image)

# Convert the list of padded images to a NumPy array
padded_input_test = np.array(padded_inputs)

# Now padded_images_array contains all the 2D images padded to the desired dimensions
print(padded_input_test.shape)

# Pad each 2D image to match the maximum width and height
padded_inputs = []
for image in sliced_output_test:
    pad_width = ((0, padded_height - image.shape[0]), (0, max_width - image.shape[1]))
    padded_image = np.pad(image, pad_width, mode='constant')
    padded_inputs.append(padded_image)

# Convert the list of padded images to a NumPy array
padded_output_test = np.array(padded_inputs)

#the final shapes of images
array_x_shape=padded_output_test.shape[1]
array_y_shape=padded_output_test.shape[2]

# Now padded_images_array contains all the 2D images padded to the desired dimensions
print(padded_output_test.shape)

batch_size=32

class ResNetPretrainedModel(nn.Module):
    def __init__(self, num_classes=4, device="cuda:0"):
        super(ResNetPretrainedModel, self).__init__()
        # Load the pretrained ResNet model
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)

        # Modify the first convolutional layer to accept 1 channel
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Modify the final fully connected layer for the number of classes
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes * array_x_shape * array_y_shape)  # Adjusted for one-hot encoding

        self.resnet = resnet
        self.num_classes = num_classes
        self.device = device
        self.to(device)  # Move the model to the specified device

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, array_x_shape, array_y_shape, self.num_classes)
        return x

    def training_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)

        loss = self.calculate_loss(outputs, targets)
        return loss

    def on_train_epoch_end(self, train_loss):
        print(f'Train Loss: {train_loss}')

    def test_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)

        loss = self.calculate_loss(outputs, targets)
        return {"test_loss": loss}

    def calculate_loss(self, outputs, targets):
        # Apply argmax along the last dimension
        targets = torch.argmax(targets, dim=3)
        predictions = F.softmax(outputs, dim=3).permute(0,3,1,2)

        # Convert to floating-point dtype
        predictions_float = predictions.float().requires_grad_()
        targets_float = targets.long()

        # Initialize the Dice metric
        dice_metric = DiceLoss(mode='multiclass', classes=[0,1,2,3], ignore_index=0)

        # Compute the Dice loss
        dice_loss = dice_metric(predictions_float, targets_float)

        return dice_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=5e-2)

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return test_loader


class CustomDataset(Dataset):
    def __init__(self, inputs, labels, num_classes=4):
        self.inputs = inputs
        self.labels = labels
        self.num_classes = num_classes

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        label_data = self.labels[idx]

        if not torch.is_tensor(input_data):
            input_data = torch.from_numpy(input_data).float()

        if not torch.is_tensor(label_data):
            label_data = torch.from_numpy(label_data).long()

        return input_data, label_data


# Normalize the inputs to the range [0, 1]
padded_input_train_normalized = (padded_input_train - np.min(padded_input_train)) / (np.max(padded_input_train) - np.min(padded_input_train))
padded_input_val_normalized = (padded_input_val - np.min(padded_input_val)) / (np.max(padded_input_val) - np.min(padded_input_val))
padded_input_test_normalized = (padded_input_test - np.min(padded_input_test)) / (np.max(padded_input_test) - np.min(padded_input_test))

# Add a channel dimension to the input data
padded_input_train_normalized = padded_input_train_normalized[:, np.newaxis, :, :]
padded_input_val_normalized = padded_input_val_normalized[:, np.newaxis, :, :]
padded_input_test_normalized = padded_input_test_normalized[:, np.newaxis, :, :]

# One-hot encode the labels
padded_output_train_one_hot = F.one_hot(torch.from_numpy(padded_output_train).long(), num_classes=4).float()
padded_output_val_one_hot = F.one_hot(torch.from_numpy(padded_output_val).long(), num_classes=4).float()
padded_output_test_one_hot = F.one_hot(torch.from_numpy(padded_output_test).long(), num_classes=4).float()

# Stack the datasets
train_dataset = CustomDataset(padded_input_train_normalized, padded_output_train_one_hot)
val_dataset = CustomDataset(padded_input_val_normalized, padded_output_val_one_hot)
test_dataset = CustomDataset(padded_input_test_normalized, padded_output_test_one_hot)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

# Initialize the model
model = ResNetPretrainedModel(device="cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# Training loop
max_epochs = 100
optimizer = optim.Adam(model.parameters(), lr=5e-3)

# Add early stopping
best_val_loss = float('inf')
patience = 5  # Number of epochs to wait for improvement
wait_count = 0

for epoch in range(max_epochs):
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.calculate_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)
    model.on_train_epoch_end(average_train_loss)

    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = model.calculate_loss(outputs, targets)
            val_loss += loss.item()

        average_val_loss = val_loss / len(val_loader)

        # Early stopping and model checkpointing
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            wait_count = 0

            # Save the model checkpoint
            checkpoint_path = "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_path)
        else:
            wait_count += 1
            if wait_count >= patience:
                print("Early stopping. No improvement in validation loss.")
                break
    print("Validation Loss:", average_val_loss)


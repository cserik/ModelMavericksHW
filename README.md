# ModelMavericksHW

This repository was created by students of the Budapest University of Technology and Economics to fulfill the homework requirements of the Deep Learning subject.

## Team name
Model Mavericks

## Team members
Egyedi Zsolt - I9D6EJ\
Rimai Dániel - BR2BUJ\
Csató Erik - IRKR10

## Our project
### Description
In this project, you'll dive into the idea of using multiple models together, known as model ensembles, to make our deep learning solutions more accurate. They are a reliable approach to improve accuracy of a deep learning solution for the added cost of running multiple networks. Using ensembles is a trick that's widely used by the winners of AI competitions. Task of the students: explore approaches to model ensemble construction for semantic segmentation, select a dataset (preferentially cardiac MRI segmentation, but others also allowed), find an open-source segmentation solution as a baseline for the selected dataset and test it. Train multiple models and construct an ensemble from them. Analyse the improvements, benefits and added costs of using an ensemble. 

### Dataset
We use the following dataset: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb

### Milestone 1
Notebook with some plots: *Model_Mavericks.ipynb*

#### Containerization

The files needed to build the container are located in the milestone1 folder. The Dockerfile helps to build the container, the requirements.txt specifies the packages (and the appropriate versions) that we used. The data_preprocessing.py script helps to preprocess the data and load it in the right format.

##### Building and Running the Docker Container

To run the Model Mavericks Preprocessor in a Docker container, follow these steps:

1. **Build the Docker Image:**

   Use the following command in the milestone1 folder to build the Docker image. This will create an image with the name `model-mavericks-preprocessor`:

   ```bash
   docker build -t model-mavericks-preprocessor .

2. **Run the Docker Image:**

   Once the image is built, you can run the Docker container using the following command:

   ```bash
   docker run -it model-mavericks-preprocessor

### Milestone 3
 - Notebook of our baseline model: *baseline_v1.ipynb*
 - Notebook of our U-Net architecture: *TriUnet.ipynb*
 - Statistics of the dataset we used and some data augmentation examples: *data_statistics.ipynb*

#### Containerization

The files needed to build the container are located in the milestone3/docker_baseline_model folder. The Dockerfile helps to build the container, the requirements.txt specifies the packages (and the appropriate versions) that we used. The train.py script is responsible for training the baseline model.

##### Building and Running the Docker Container

To start the training of the baseline model in a Docker container, follow these steps:

1. **Build the Docker Image:**

   Use the following command in the milestone3/docker_baseline_model folder to build the Docker image. This will create an image with the name `train-baseline-model`:

   ```bash
   docker build -t train-baseline-model .

2. **Run the Docker Image on CPU:**

   Once the image is built, you can run the Docker container using the following command:

   ```bash
   docker run -it train-baseline-model

3. **Run the Docker Image on GPU(if available):**
   
   Replace the Dockerfile in the milestone3/docker_baseline_model folder with the one located in the milestone3/dockerfile_cuda directory and rebuild the container according the first step (see above). Once the image is built, you can run the Docker container using the following command:

   ```bash
   docker run --gpus all -it train-baseline-model

#### User Interface

The user interface consists of two main panels. Using the interface on the left, you can upload a file containing an MRI recording. The file format can be one of the following extensions: jpeg, jpg, png, nii, nii.gz. Using the file given as input, the neural network will determine the extent of heart diseases. The predicted area will be colored on the input image, and then it will appear in the right panel. By pressing the left button, the selected file is deleted and it is possible to select a new file.

### Related Works

- [https://nipy.org/nibabel/gettingstarted.html](https://nipy.org/nibabel/gettingstarted.html)
- [https://nipy.org/nibabel/nibabel_images.html](https://nipy.org/nibabel/nibabel_images.html)
- [https://ieeexplore.ieee.org/abstract/document/8360453](https://ieeexplore.ieee.org/abstract/document/8360453)
- [https://nipy.org/nibabel/coordinate_systems.html](https://nipy.org/nibabel/coordinate_systems.html)


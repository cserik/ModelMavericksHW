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

####Containerization

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

### Related works
https://nipy.org/nibabel/gettingstarted.html
https://nipy.org/nibabel/nibabel_images.html
https://ieeexplore.ieee.org/abstract/document/8360453
https://nipy.org/nibabel/coordinate_systems.html

# Use an official CUDA-enabled image as a parent image
FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04 

# Rest of your Dockerfile
WORKDIR /app

# Install required packages
RUN apt-get update && \
    apt-get install -y python3.8 python3.8-dev python3-pip wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Create a symbolic link to make Python 3.8 the default python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Upgrade pip for Python 3.8
RUN python -m pip install --upgrade pip

# Get MNIST dataset
RUN wget -O database.zip https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download

# Unzip the database.zip file
RUN unzip database.zip

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64
ENV PATH=$PATH:$CUDA_HOME/bin

# Run the preprocessing script when the container launches
CMD ["python", "train.py"]

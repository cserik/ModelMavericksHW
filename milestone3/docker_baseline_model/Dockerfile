# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory to /app
WORKDIR /app

RUN pip install --upgrade pip

# Get MNIST dataset
RUN wget -O database.zip https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download

# Unzip the database.zip file
RUN unzip database.zip

# Install any needed packages specified in requirements.txt
# Copy the current directory contents into the container at /app
COPY . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Run the preprocessing script when the container launches
CMD ["python", "train.py"]

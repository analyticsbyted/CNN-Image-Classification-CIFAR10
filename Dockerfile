# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container at /app
COPY app.py .
COPY preprocessing.py .
COPY model/cifar10_model.keras model/cifar10_model.keras

# Expose the port that Gradio runs on
EXPOSE 7860

# Run the Gradio application
CMD ["python", "app.py"]

# Project Overview

This project is a deep learning-based image classification system. It utilizes a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 distinct classes. The primary technologies used are Python, TensorFlow/Keras for model development, and various other libraries for data manipulation and visualization.

# Building and Running

## Dependencies

While a `requirements.txt` file is not present, the following dependencies can be inferred from the project files:

*   tensorflow
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn
*   Pillow
*   tabulate

You can install these using pip:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn Pillow tabulate
```

## Running the Project

The main workflow of the project, including data loading, preprocessing, model training, and evaluation, is contained within the `CNN_Image_Classification_CIFAR10.ipynb` Jupyter Notebook. To run the project, you can open and execute the cells in this notebook.

## Prediction

To perform predictions on new images, you can use the `prediction.py` script. The script takes an image and a trained model as input and outputs the predicted class and confidence score.

Example usage:

```python
from PIL import Image
from tensorflow.keras.models import load_model
from prediction import predict_image

# Load the trained model
model = load_model('model/')

# Load and preprocess the image
image = Image.open('images/picture4.jpg')

# Make a prediction
predicted_class, confidence_score = predict_image(image, model)

# Print the result
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f"Predicted class: {class_labels[predicted_class]}")
print(f"Confidence score: {confidence_score}")
```

# Development Conventions

*   The code is primarily written in Python.
*   The main development and experimentation environment is Jupyter Notebook.
*   The code is well-documented with comments and markdown explanations in the notebook.
*   The project follows a standard structure for a deep learning project, with separate files for preprocessing, prediction, and a directory for the saved model.

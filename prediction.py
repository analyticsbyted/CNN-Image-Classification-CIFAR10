from tensorflow.keras.models import load_model
import numpy as np
from preprocessing import preprocess_image
from PIL import Image
import os

def predict_image(image_path, model_path='model/'):
    model = load_model(model_path)
    image = Image.open(image_path)
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    confidence_score = np.max(predictions[0])
    return predicted_class, confidence_score

if __name__ == '__main__':
    # Get a list of all image files in the images directory
    image_files = [f for f in os.listdir('images') if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Predict for each image
    for image_file in image_files:
        image_path = os.path.join('images', image_file)
        predicted_class, confidence_score = predict_image(image_path)
        class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        print(f"Image: {image_file}, Predicted class: {class_labels[predicted_class]}, Confidence: {confidence_score:.2f}")
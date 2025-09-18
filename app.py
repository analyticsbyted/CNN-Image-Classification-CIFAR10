import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model/cifar10_model.keras')

# Define the class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image):
    # The input image is a NumPy array, convert it to a PIL Image
    # so we can resize it easily.
    pil_image = Image.fromarray(image.astype('uint8'), 'RGB')

    # Preprocess the image to match the model's input requirements
    # 1. Resize to 32x32 pixels
    img = pil_image.resize((32, 32))
    # 2. Convert to a numpy array
    img_array = np.array(img)
    # 3. Normalize pixel values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    # 4. Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)

    # Format the output for Gradio's Label component
    # It expects a dictionary of class names to confidence scores
    confidences = {class_labels[i]: float(predictions[0][i]) for i in range(10)}
    
    return confidences

# Create the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload an Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="CIFAR-10 Image Classifier",
    description="Upload an image of an airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck to see the model's prediction.",
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()

import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model('model/cifar10_model.keras')

# Define the class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image):
    # The input image is a NumPy array, convert it to a PIL Image
    pil_image = Image.fromarray(image.astype('uint8'))

    # Preprocess the image to match the model's input requirements
    img = pil_image.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)

    # Format the output for Gradio's Label component
    confidences = {class_labels[i]: float(predictions[0][i]) for i in range(10)}
    
    return confidences

# Define example images
# Assuming example images are in the 'images' directory
example_images_dir = "images"
example_images = [
    os.path.join(example_images_dir, f) for f in os.listdir(example_images_dir)
    if f.endswith(('.png', '.jpg', '.jpeg'))
]

# Create the Gradio interface using Blocks for more control
with gr.Blocks(theme=gr.themes.Soft(), title="CIFAR-10 Image Classifier") as demo:
    gr.Markdown("# CIFAR-10 Image Classifier")
    gr.Markdown("Upload an image of an airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck to see the model's prediction.")

    with gr.Row():
        image_input = gr.Image(label="Upload an Image")
        label_output = gr.Label(num_top_classes=3, label="Predictions")

    gr.Examples(
        examples=example_images,
        inputs=image_input,
        outputs=label_output,
        fn=predict,
        cache_examples=True,
    )

    image_input.change(fn=predict, inputs=image_input, outputs=label_output)

    gr.Markdown("\n---") # Separator
    gr.Markdown("© 2025 Ted Dickey. All rights reserved.")

# Launch the app
if __name__ == "__main__":
    demo.launch()
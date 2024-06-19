import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2

# Function to check if a directory exists and create it if not
def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load trained generator model
model_name = "GAN_2"
model_directory = f"./{model_name}"
g = load_model(f"{model_directory}/model/g_model.h5")

# Function to generate images
def generate_images(n_images=10, z_dim=100):
    # Generate random noise
    noise = np.random.normal(0, 1, size=(n_images, z_dim))
    # Generate images from noise
    generated_images = g.predict(noise)
    # Reshape images to 28x28
    generated_images = generated_images.reshape(n_images, 28, 28)
    return generated_images

# Function to plot generated images
def plot_generated_images(generated_images, n_cols=5):
    n_images = len(generated_images)
    n_rows = (n_images - 1) // n_cols + 1
    plt.figure(figsize=(2*n_cols, 2*n_rows))
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.show()

# Function to save generated images
def save_generated_images(generated_images, output_directory):
    check_directory(output_directory)
    existing_files = len([name for name in os.listdir(output_directory)])
    for i, image in enumerate(generated_images):
        image = image * 255
        # Convert grayscale image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Save RGB image
        cv2.imwrite(f"{output_directory}/{existing_files + i + 1}.jpg", rgb_image)


# Check if the model directory exists
if not os.path.exists(model_directory):
    raise FileNotFoundError(f"Model directory '{model_directory}' does not exist.")

# Check if the 'image' directory exists within the model directory
image_directory = f"{model_directory}/image"
check_directory(image_directory)

# Check if the 'test' directory exists within the 'image' directory
test_directory = f"{model_directory}/test"
check_directory(test_directory)

# Generate images
generated_images = generate_images(n_images=10)

plot_generated_images(generated_images)

# Save generated images to the 'test' directory
save_generated_images(generated_images, test_directory)

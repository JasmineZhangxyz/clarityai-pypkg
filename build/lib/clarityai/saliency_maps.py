import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

class SaliencyMapGenerator:
    """
    Creates saliency map for an image put through a CNN. Useful in visualizing which 
    parts of an image the CNN is focusing on when making its prediction.
    """
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def preprocess_image(self, img):
        input_image = tf.expand_dims(tf.cast(img, tf.float32), axis=0)
        input_image = input_image / 255.0
        return input_image

    def generate_saliency_map(self, img):
        # Preprocess the image
        test_image = self.preprocess_image(img)

        # Calculate the saliency map
        with tf.GradientTape() as tape:
            tape.watch(test_image)
            predictions = self.model(test_image)
            top_prediction = tf.argmax(predictions[0])
            top_score = predictions[0, top_prediction]
        
        # Calculate the gradients of the top score with respect to the input image
        gradients = tape.gradient(top_score, test_image)[0]
        saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)

        # Normalize the saliency map to the range [0, 1]
        saliency_map = saliency_map / tf.reduce_max(saliency_map)

        # Convert the saliency map and original image to numpy arrays
        saliency_map = saliency_map.numpy()

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the original image
        axs[0].imshow(img)
        axs[0].set_title('Original Image')

        # Plot the saliency map
        saliency_img = axs[1].imshow(saliency_map, cmap='coolwarm')
        axs[1].set_title('Saliency Map')

        # Create a ScalarMappable object using the saliency map
        saliency_colorbar = fig.colorbar(saliency_img, ax=axs[1])

        # Display the figure
        plt.show()
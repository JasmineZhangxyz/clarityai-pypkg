import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

class AttentionMapGenerator:
    """
    Creates attention maps for layers of a CNN. Useful in visualizing which 
    parts of an image the model is focusing on when making its prediction.
    """
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def preprocess_image(self, img):
        input_image = tf.expand_dims(tf.cast(img, tf.float32), axis=0)
        input_image = input_image / 255.0
        return input_image

    def get_activations_at(self, input_image, i):
        out_layer = self.model.layers[i]
        inputs, outputs = self.model.inputs, out_layer.output
        model = tf.keras.models.Model(inputs, outputs)
        return model.predict(input_image)

    def postprocess_activations(self, activations):
        output = np.abs(activations)
        output = np.sum(output, axis=-1).squeeze()
        output = cv2.resize(output, (128, 128))
        output /= output.max()
        output *= 255
        return 255 - output.astype('uint8')

    def apply_heatmap(self, weights, img):
        heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
        return heatmap

    def generate_heatmaps(self, img, layer_indices):
        input_image = self.preprocess_image(img)
        heatmaps = []
        for i in layer_indices:
            activations = self.get_activations_at(input_image, i)
            weights = self.postprocess_activations(activations)
            heatmap = self.apply_heatmap(weights, img)
            heatmaps.append(heatmap)
        return heatmaps

    def plot_heatmaps(self, heatmaps):
        level_maps = np.concatenate(heatmaps, axis=1)
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        ax = plt.imshow(level_maps)
        plt.show()
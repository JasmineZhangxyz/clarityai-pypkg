import unittest
import numpy as np
import cv2

from clarityai.attention_maps import AttentionMapGenerator

class TestAttentionMapGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = AttentionMapGenerator("examples/my_model.h5")

    def test_preprocess_image(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        preprocessed_img = self.generator.preprocess_image(img)
        self.assertEqual(preprocessed_img.shape, (1, 128, 128, 3))
        self.assertTrue(np.allclose(preprocessed_img, np.zeros((1, 128, 128, 3), dtype=np.float32)))

    def test_postprocess_activations(self):
        activations = np.random.rand(128, 128)
        processed = self.generator.postprocess_activations(activations)
        self.assertEqual(processed.shape, (128, 128))
        self.assertTrue(processed.dtype == np.uint8)

    def test_apply_heatmap(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        weights = np.random.rand(128, 128)
        heatmap = self.generator.apply_heatmap(weights, img)
        self.assertEqual(heatmap.shape, (128, 128, 3))
        self.assertTrue(heatmap.dtype == np.uint8)

    def test_generate_heatmaps(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        layer_indices = [0, 1, 2]       # can replace with other layer indices
        heatmaps = self.generator.generate_heatmaps(img, layer_indices)
        self.assertEqual(len(heatmaps), len(layer_indices))
        for heatmap in heatmaps:
            self.assertEqual(heatmap.shape, (128, 128, 3))
            self.assertTrue(heatmap.dtype == np.uint8)

if __name__ == '__main__':
    unittest.main()

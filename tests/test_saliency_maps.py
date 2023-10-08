import unittest
import numpy as np
import cv2

from clarityai.saliency_maps import SaliencyMapGenerator

class TestSaliencyMapGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = SaliencyMapGenerator("examples/my_model.h5")

    def test_preprocess_image(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        preprocessed_img = self.generator.preprocess_image(img)
        self.assertEqual(preprocessed_img.shape, (1, 128, 128, 3))
        self.assertTrue(np.allclose(preprocessed_img, np.zeros((1, 128, 128, 3), dtype=np.float32)))

    def test_generate_saliency_map(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        saliency_map = self.generator.generate_saliency_map(img)

        # Check the saliency map is a numpy array
        self.assertTrue(isinstance(saliency_map, np.ndarray))

        # Check the shape of the saliency map
        self.assertEqual(saliency_map.shape, (128, 128))

        # Check the range of values in the saliency map (between 0 and 1)
        self.assertTrue(np.min(saliency_map) >= 0)
        self.assertTrue(np.max(saliency_map) <= 1)

if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np

import geogen.model as geo
import geogen.plot as geovis


class TestGeoModel(unittest.TestCase):

    def test_initialization(self):
        """Test initialization with various bounds and resolutions."""
        model = geo.GeoModel(bounds=(0, 10), resolution=10)
        self.assertEqual(model.bounds, ((0, 10), (0, 10), (0, 10)))
        self.assertEqual(model.resolution, (10, 10, 10))

        model = geo.GeoModel(bounds=((0, 10), (0, 20), (0, 30)), resolution=(5, 10, 15))
        self.assertEqual(model.bounds, ((0, 10), (0, 20), (0, 30)))
        self.assertEqual(model.resolution, (5, 10, 15))

    def test_invalid_bounds(self):
        """Test initialization with invalid bounds."""
        with self.assertRaises(AssertionError):
            geo.GeoModel(bounds=(0, 1, 2))  # Incorrect bounds tuple

    def test_invalid_resolution(self):
        """Test initialization with invalid resolution values."""
        with self.assertRaises(ValueError):
            geo.GeoModel(bounds=(0, 10), resolution="high")  # Non-integer resolution

        with self.assertRaises(AssertionError):
            geo.GeoModel(bounds=(0, 10), resolution=(10, 10))  # Tuple not of length 3

    def test_mesh_setup(self):
        """Test the mesh grid setup based on provided bounds and resolution."""
        resolution = (3, 5, 7)
        model = geo.GeoModel(bounds=((0, 1), (0, 1), (0, 1)), resolution=resolution)
        model._setup_mesh()
        self.assertEqual(model.X.shape, resolution)
        self.assertEqual(model.Y.shape, resolution)
        self.assertEqual(model.Z.shape, resolution)
        prod = np.prod(resolution)
        self.assertEqual(len(model.xyz), prod)  # Check if XYZ is correctly flattened


if __name__ == "__main__":
    unittest.main()

import numpy as np


def _parse_bounds(bounds):
    """Ensure bounds are in the form of ((x_min, x_max), (y_min, y_max), (z_min, z_max))."""
    if isinstance(bounds[0], tuple):
        assert len(bounds) == 3 and all(len(b) == 2 for b in bounds), "Invalid bounds format."
    elif isinstance(bounds, tuple) and len(bounds) == 2:
        bounds = (bounds, bounds, bounds)
    else:
        raise ValueError("Bounds must be a tuple of 2 values or a tuple of three 2-tuples.")
    return bounds


def _rng_method(rng, name):
    return getattr(np.random if rng is None else rng, name)


def random_point_in_ellipsoid(bounds, rng=None):
    """Generate a random point within an ellipsoid defined by bounds on x, y, z axes."""

    # Parse bounds and calculate centers and radii
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = _parse_bounds(bounds)
    x_radius = (x_max - x_min) / 2
    y_radius = (y_max - y_min) / 2
    z_radius = (z_max - z_min) / 2
    center_x = x_min + x_radius
    center_y = y_min + y_radius
    center_z = z_min + z_radius

    # Random angles and radius for a unit sphere
    uniform = _rng_method(rng, "uniform")
    phi = uniform(0, 2 * np.pi)  # Azimuthal angle
    theta = uniform(0, np.pi)  # Polar angle
    u = uniform(0, 1)  # Radius
    r = u ** (1 / 3)

    # Random point in unit sphere scaled to fit the ellipsoid
    x = r * np.sin(theta) * np.cos(phi) * x_radius + center_x
    y = r * np.sin(theta) * np.sin(phi) * y_radius + center_y
    z = r * np.cos(theta) * z_radius + center_z

    return x, y, z


def random_point_in_box(bounds, rng=None):
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = _parse_bounds(bounds)
    uniform = _rng_method(rng, "uniform")
    x_loc = uniform(x_min, x_max)
    y_loc = uniform(y_min, y_max)
    z_loc = uniform(z_min, z_max)
    return x_loc, y_loc, z_loc


def random_angle_degrees(rng=None):
    """Generate a random angle in degrees from 0 to 360."""
    return _rng_method(rng, "uniform")(0, 360)


def log_normal_params(mean, std_dev):
    """Calculate the parameters of a log-normal distribution from the mean and standard deviation."""
    mu = np.log(mean**2 / np.sqrt(std_dev**2 + mean**2))
    sigma = np.sqrt(np.log(1 + std_dev**2 / mean**2))
    return mu, sigma


def beta_min_max(a, b, min_val, max_val, rng=None):
    """Generate a beta distributed random number with specified min and max values."""
    return min_val + (max_val - min_val) * _rng_method(rng, "beta")(a, b)

import functools

import numpy as np
from scipy.ndimage import gaussian_filter1d


def damped_fourier_wave_fun(n_cycles, num_harmonics, frequency, amplitudes, phases, rms_scale):
    result = np.zeros_like(n_cycles)
    for n, (amplitude, phase) in enumerate(zip(amplitudes, phases), start=1):
        result += amplitude * np.sin(2 * np.pi * frequency * n * n_cycles + phase)
    return result * rms_scale


class FourierWaveGenerator:
    """
    Generates a Fourier series function with given number of harmonics.
    Amplitudes decrease with frequency and the total RMS is normalized
    to sqrt(2)/2 like a normal sine wave. This gives an overall appearance
    of amplitude of roughly 1 like a regular sine wave.

    Parameters
    ----------
    num_harmonics : int
        Number of harmonics to generate.
    frequency : float, optional
        Frequency of the wave, by default 1.
    smoothness : float, optional
        Exponent of the amplitude decay with frequency, by default 1.0.

    Methods
    -------
    generate()
        Generate a random Fourier series function f(n_cycles: np.ndarray) -> np.ndarray
    """

    def __init__(self, num_harmonics, frequency=1, smoothness=1.0):
        self.num_harmonics = num_harmonics
        self.frequency = frequency
        self.smoothness = smoothness

    def generate(self):
        """
        Generate a random Fourier series function.

        The returned function has the signature:
            f(n_cycles: np.ndarray) -> np.ndarray

        where `n_cycles` is an array of the number of cycles to generate.
        """

        amplitudes = []
        phases = []
        total_power = 0
        order = self.smoothness
        for n in range(1, self.num_harmonics + 1):
            amplitude = np.random.normal(loc=1.0 / (n**order), scale=0.5 / (n**order))
            amplitude = abs(amplitude)  # Ensure non-negative amplitude
            phase = np.random.uniform(0, 2 * np.pi)
            amplitudes.append(amplitude)
            phases.append(phase)
            total_power += amplitude**2

        rms_scale = np.sqrt(1 / total_power)
        amplitudes = np.array(amplitudes)
        phases = np.array(phases)

        return functools.partial(
            damped_fourier_wave_fun,
            num_harmonics=self.num_harmonics,
            frequency=self.frequency,
            amplitudes=amplitudes,
            phases=phases,
            rms_scale=rms_scale,
        )

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


def noisy_sine_wave(frequency=1, smoothing=20, noise_scale=0.1):
    # Noisy sine
    def noisy_sin_wave_func(n_cycles):
        # Deterministic sinusoidal component
        deterministic = np.sin(2 * np.pi * frequency * n_cycles)

        # Generate random noise
        random_noise = np.random.normal(scale=noise_scale, size=n_cycles.shape)

        # Smooth the random noise
        smoothed_noise = gaussian_filter1d(random_noise, sigma=smoothing)

        # Combine the deterministic and random parts
        return deterministic + smoothed_noise

    return noisy_sin_wave_func

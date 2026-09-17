import numpy as np

n = 19656
delta = 1 / n
# N and M are assumed to N+M < n/2
# Time is assumed to be [0, 2pi]

def discrete_fourier_coeffs(dX: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the discrete Fourier coefficients of a stochastic process as in Def. 5.4 in page 14.
    Needs a phase correction because FFT is computed starting at j=0, instead of the definition's j=1.
    Returns ordered Fourier coefficients.
    """
    
    if dX.ndim == 1:
        dX = dX[:, None]
    
    raw_coeffs = np.fft.fft(dX, axis=0)
    k = np.fft.fftfreq(n, d=delta)

    phase_correction = np.exp(-2j * np.pi * k / n)[:, None]
    corrected_coeffs = phase_correction * raw_coeffs

    ordered_coeffs = np.fft.fftshift(corrected_coeffs, axes=0)
    ordered_freqs = np.fft.fftshift(k)

    coeffs = 1/(2*np.pi) * ordered_coeffs

    return (coeffs, ordered_freqs)


def malliavin_mancino_theorem(dX: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the variance coefficients according to the Malliavin-Mancino theorem, page 17.
    Uses the convolution theorem + FFT for an efficient computation.
    """

    coeffs, frequencies = discrete_fourier_coeffs(dX)

    indicator = np.abs(frequencies) <= N
    c_s = coeffs * indicator[:, None]

    # Linear convolution
    n_fft = 65536
    c_s_fft = np.fft.fft(c_s, n_fft, axis=0)
    coeffs_fft = np.fft.fft(coeffs, n_fft, axis=0)

    product = c_s_fft * coeffs_fft
    convolution = np.fft.ifft(product, axis=0)

    # Realign convolution & compute coefficients
    convolution = convolution[n//2 : 3*n//2, :]
    c_v = 2*np.pi / (2*N + 1) * convolution

    return (c_v, frequencies)
    

def fourier_estimator(dX: np.ndarray, N: int, M: int) -> np.ndarray:
    """
    Computes the Fourier-based estimator obtained via the Malliavin-Mancino theorem and the Fejèr kernel.
    """

    coeffs, frequencies = malliavin_mancino_theorem(dX, N)
    indicator = np.abs(frequencies) <= M
    fejer = (1 - np.abs(frequencies) / (M + 1)) * indicator

    weighted_coeffs = fejer[:, None] * coeffs

    # Compute the estimator using numpy's original order and removing the ifft default scaling it adds
    unordered_coeffs = np.fft.ifftshift(weighted_coeffs, axes=0)
    estimator = np.fft.ifft(unordered_coeffs, axis=0) * n 

    estimator = np.roll(estimator, -1, axis=0)

    # Returns a complex array. Maybe should 
    return estimator


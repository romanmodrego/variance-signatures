import numpy as np

n = 19656
delta = 1 / n
h = 3.06 * delta

def one_sided_gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """
    Computes the one-sided gaussian kernel as defined in section 7.2 (page 24)
    """
    return np.sqrt(2 / np.pi) * np.exp(-u**2 / 2) * np.where(u <= 0, 1.0, 0.0)


def gaussian_estimator(dX: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Computes the kernel-based estimator as defined in section 7.2 (page 24)
    It's O(m*n^2), do not run on big arrays. It's just for illustrative purposes.
    """
    # indexing starts at the right endpoint of the increment
    time = time[1:]
    if dX.ndim == 1:
        dX = dX[:, None]

    u = (time[:, None] - time[None, :]) / h
    kernel = one_sided_gaussian_kernel(u) 
    dX_squared = dX**2

    v_G = 1/h * kernel.T @ (dX_squared)

    return v_G[5:,:]


def efficient_gaussian_estimator(dX: np.ndarray) -> np.ndarray:
    """
    Computes the kernel-based estimator as defined in section 7.2 (page 24)
    Uses the Toeplitz structure of the kernel and the convolution theorem (using FFT) to make it more efficient.
    https://ccrma.stanford.edu/~jos/ReviewFourier/FFT_Convolution.html
    """

    if dX.ndim == 1:
        dX = dX[:, None]

    lags = np.arange(n)
    u = -lags * delta / h

    kernel = one_sided_gaussian_kernel(u)
    dX_squared = dX**2

    # To apply the FFT as a linear conv and avoid time aliasing, we zero-pad the dimensions of the vectors to at least 2n-1 = 39,311
    # Since Cooley-Tukey FFT works for powers of 2, we zero-pad to the next power of 2 = 65536
    n_fft = 65536
    kernel_fft = np.fft.rfft(kernel, n_fft)
    dX_fft = np.fft.rfft(dX_squared, n_fft, axis=0)

    product = kernel_fft[:, None] * dX_fft
    convolution = np.fft.irfft(product, n_fft, axis=0)

    v_G = 1/h * convolution[:n, :]

    # Output grid restriction to compare it to v_rect. Can be adjusted later if needed.
    return v_G[5:,:]
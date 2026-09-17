The functions described correspond to the ones seen in the script `fourier_estimator.py`, which computes the Fourier estimator for the spot variance as 

$$\hat{\nu}_M^{n,N}(t) = \sum_{|k| \leq M} \left(1 - \frac{|k|}{M+1}\right) c_k^{n,N}(\nu)e^{ikt}$$

This is done in three different steps: 
1. Compute the discrete Fourier coefficients of the process $dX$.
2. Using the Malliavin-Mancino theorem, compute the Fourier coefficients of the variance process via the scaled convolution of the coefficients in step 1 for frequencies $[-N,N]$. 
3. Calculate the variance estimator using a Fejèr sum of  the coefficients of step 2.

| Function | Description |
|---|---|
| `discrete_fourier_coeffs` | Calculates the discrete Fourier coefficients of $dX$. Applies a phase correction, since NumPy's FFT calculates the sum over $j=0,\dots, n-1$ but `dX[0]` corresponds to $\Delta X_{t_1}$. Rearranges the coefficients to appear by frequency, as $-n/2, -n/2 + 1, \dots, 0, 1, \dots,  n/2 - 1$, to avoid indexing problems when applying the Malliavin-Mancino theorem. NumPy's default ordering is $0, 1, \dots, n/2 - 1, -n/2, -n/2+1, \dots, -1$.|
| `malliavin_mancino_theorem` | Applies the Malliavin-Mancino theorem using an indicator function to truncate the convolution to $[-N,N]$. Uses the convolution theorem and FFTs for efficient computations, as seen in `one_sided_gaussian_kernel.py`. Since it is padded to a linear convolution of length $2n-1$ to avoid aliasing, the coefficients are truncated to the original $[-n/2, n/2-1]$ grid, corresponding to `[n//2 : 3n//2, :]` in matrix indexing. Returns the coefficients and frequencies. |
| `fourier_estimator` | Computes the Fejèr weights truncated to $[-M, M]$ and uses them to compute the weighted coefficients from the output of the Malliavin-Mancino theorem. Rearranges them to the default NumPy ordering to apply the iFFT and scales it to remove its default normalization by $1/n$. Keeps time alignment by rearranging the output order from $(t_0, t_1, \dots, t_n-1)$ to $(t_1, \dots, t_n)$ by moving the first term to the last position; this can be done because it is periodic. |
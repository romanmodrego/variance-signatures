The functions described correspond to the ones seen in the script `one_sided_gaussian_kernel.py`, which computes the One-sided Gaussian Kernel estimator for the spot variance as 

$$\hat{\nu}_G^h(t) = \frac{1}{h} \sum_{i=1}^n K_G\left(\frac{t_i - t}{h}\right) (\Delta X_{t_i})^2$$

| Function | Description |
|---|---|
| `one_sided_gaussian_kernel` | Calculates a one-sided Gaussian kernel, defined as $K_G(u) := 2 \phi(u)\mathbf{1}_{(-\infty, 0]}(u) = \sqrt{\frac{2}{\pi}}e^{-u^2/2}\mathbf{1}_{(-\infty, 0]}(u)$. |
| `gaussian_estimator` | Naive implementation of the Gaussian kernel estimator. While clear and mathematically correct, it is inefficient: it performs a total of $O(n^2m)$ operations. For $n = 19656$, this implies that, for every path $m$, it computes $\approx 386$ million operations. |
| `efficient_gaussian_estimator` | Efficient version of `gaussian_estimator`. Exploits the Toeplitz property and the Convolution Theorem to efficiently compute $\hat{\nu}_G^h$, employing the FFT. |

To make the code as efficient as possible, the following have been used:

- **Toeplitz matrix**: square matrix in which each diagonal (from left to right) shares the same value, that is, $a_{i,j} = a_{i+1,j+1}$. Since we have $t_j = j\delta$, then $t_i - t_j = (i - j)\delta$ and the lagged-time matrix $(t_i - t_j)_{ij} = \delta (i-j)_{ij}$ is Toeplitz. 
By the definition of $K_G(u)$, we observe that it is upper triangular and Toeplitz, given $u = \frac{t_i - t_j}{h} = \frac{(i-j)\delta}{3.06\delta} = \frac{i-j}{3.06}$. This implies that there are $n$ unique kernel values $K_G(\frac{t_i - t_j}{h})$. This can be exploited to save computations by just calculating the kernel for $n$ lags.

- **Convolution Theorem**: for $f,g$ smooth, $2\pi$-periodic functions on $[0,2\pi]$, their Fourier coefficients satisfy $c_k(fg) = \sum_{s\in \mathbb{Z}}c_s(f)c_{k-s}(g)$. If $\mathcal{F}$ denotes the Fourier transform, then $\mathcal{F}[fg] = \mathcal{F}[f] * \mathcal{F}[g]$. Thus, $fg = \mathcal{F}^{-1}(\mathcal{F}[f] * \mathcal{F}[g])$. Moreover, the discrete Fourier transform satisfies $\mathcal{F}[f*g] = \mathcal{F}[f]\mathcal{F}[g]$. Since $\hat{\nu}_G^h$ is a convolution, we can use the FFT for this second part of the convolution theorem to compute it efficiently.

- **Circular and linear convolutions**: see https://ccrma.stanford.edu/~jos/ReviewFourier/FFT_Convolution.html for details.



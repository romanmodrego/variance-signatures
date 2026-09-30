import numpy as np

day = 78
year = 252
obs = day * year
delta = 1 / obs


def compute_bounds(hat_nu, nu, epsilon, l):
    """
    Calculates the indexing (over 19656 obs) of the bounds for a provided estimator.
    
    """

    # Align true path with estimator
    nu = nu[6:, :]                                                               # (s - 6, nMC)

    # Calculate the MSE for each t (mean for path axis, instead of time)
    error = np.mean((hat_nu - nu)**2, axis=1)                                    # (s - 6, )
    # Index of t = 0.25 and t = 0.75
    q1 = int(len(error) * 0.25)
    q3 = int(len(error) * 0.75)

    error_int = np.median(error[q1:q3])
    error = error / error_int

    # Define |R(s) - 1| <= epsilon and look at windows of size (t+l) - t + 1 = l + 1 satisfying it
    # The cumulative sum shows how many elements up to t satisfy condition <= epsilon
    condition = np.abs(error - 1) <= epsilon
    cumulative_sum = np.concatenate(([0], np.cumsum(condition))) # = number of trues up to position j 

    # Then, to obtain the windows of size l+1 where each element satisfies the condition, we define
    L = int(round(l/delta)) # = 6
    window = L + 1
    window_counts = cumulative_sum[window:] - cumulative_sum[:-window] # = number of trues between j and j + L
    valid_windows = np.flatnonzero(window_counts == window)  # window is valid <=> #trues == L+1; returns start of valid windows

    if valid_windows.size == 0: 
        return "No valid windows in which the condition is met"
    
    lower_bound = valid_windows[0] + 6
    upper_bound = obs - (valid_windows[-1] + 6 + L)

    return lower_bound, upper_bound

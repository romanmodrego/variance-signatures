import numpy as np

day = 78
year = 252
obs = day * year
delta = 1 / obs

def MISE(nu, hat_nu):
    """
    Computes the MISE between an estimated variance path and the actual path
    Accounts for the start of the estimator at t_6 to properly define the interval.
    Uses trapezoidal integration for more accuracy.
    Returns both MISE and ISE.
    """

    # I_0 = [78/19656, 19578/19656] = [day/obs, (obs - day)/obs]; length is 250/252 = year - 2 days
    # Since hat_nu start at t_6 = 6/19656, the indexing shifts -6 relative to the original. 
    # Add 1 because Python slicing excludes upper endpoint
    hat_nu = hat_nu[day - 6 : obs - day - 6 + 1, :]                         
    nu = nu[day : obs - day + 1, :]                                         # (obs - 2*day + 1, nMC)

    squared_error = (hat_nu - nu)**2                                        # (obs - 2*day + 1, nMC)

    # Change to xp when added to class
    ise = np.trapezoid(squared_error, dx=delta, axis=0)                     # (nMc,)
    mise = np.mean(ise)                                                     # scalar

    return mise, ise
import numpy as np
from scipy.integrate import quad

hubble_const = 2.3e-10   # in 1/pc
Omega_0_m = 0.3111
Omega_0_L = 0.6889

def HubbleLaw(d_lum):
    """
    The simple Hubble Law relating the luminosity distance to the redshift
    
    Parameters:
        d_lum : float or array_like
            The luminosity distance
    
    Returns:
        out : float or array_like
            The redshift
    """
    return hubble_const * d_lum

def HubbleParameter(z):
    """
    Calculates the Hubble parameter at a given redshift in a universe with matter and dark energy content
        H = dln a / dt
    
    Parameters:
        z : float or array_like
            The redshift
    
    Returns:
        out : float or array_like
            The Hubble Parameter
    """
    return hubble_const * np.sqrt(Omega_0_L)/ np.tanh(np.arcsinh(np.sqrt(Omega_0_L/Omega_0_m/(1.+z)**3) ))

def CriticalDensity(z):
    """
    Calculates the critical density at a given redshift depending on the Hubble Parameter
        rho_crit = 3/8\pi H^2 
    
    Parameters:
        z : float or array_like
            The redshift
    
    Returns:
        out : float or array_like
            The critical density
    """
    H = HubbleParameter(z)
    return  3.*H**2 / 8. / np.pi

def Omega_m(z):
    """
    Calculates the matter density parameter at a given redshift in a universe with matter and dark energy content
        Omega_m = 1 - Omega_Lambda
    
    Parameters:
        z : float or array_like
            The redshift
    
    Returns:
        out : float or array_like
            The matter density parameter
    """
    rho_crit = CriticalDensity(z)
    return 1. - 3*hubble_const**2 * Omega_0_L / (8. * np.pi * rho_crit)


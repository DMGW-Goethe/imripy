import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad
import imripy.merger_system as ms
from scipy.special import jv


def h_2(sp, t, omega_s, R, dbg=False, acc=1e-13):
    """
    This function calculates the gravitational waveform h_+,x according to eq (25) in https://arxiv.org/pdf/1408.3534.pdf

    Parameters:
        sp (SystemProp) : The object describing the properties of the inspiralling system
        t  (array_like) : The time steps of the system evolution
        omega_s (array_like) : The corresponding orbital frequencies at the time steps
        R  (array_like) : The corresponding radii at the time steps
        dbg (bool)      : A parameter describing changing the returned variables
        acc    (float)  : An accuracy parameter that is passed to the integration function

    Returns:
    For dbg = False
        f_gw : np.ndarray
            The frequencies of the gravitational wave emission at the corresponding time steps
        h : np.ndarray
            The amplitude of the waveform in the fourier domain at the corresponding time steps
        Psi : np.ndarry
            The phase of the waveform in the fourier domain at the corresponding time steps

    For dbg = True
        f_gw : np.ndarray
            The frequencies of the gravitational wave emission at the corresponding time steps
        h : np.ndarray
            The amplitude of the waveform in the fourier domain at the corresponding time steps
        Psi : np.ndarray
            The phase of the waveform in the fourier domain at the corresponding time steps
        t_of_f : scipy.interp1d
            The interpolation object that maps time and frequency in the inspiral frame
        PhiTile : np.ndarray
            The phase that is left to observe at a given time step
        A   : np.ndarray
            The amplitude of the waveform in over time

    TODO:
        - Include computation of h_cross
    """
    # First, obtain mapping of gw frequency and time
    f_gw = omega_s / np.pi
    t_of_f = interp1d(f_gw, t, kind='cubic', bounds_error=False, fill_value='extrapolate')

    # Next, get the accumulated phase Phi
    omega_gw= UnivariateSpline(t, 2*omega_s, ext=1, k=5 )
    Phi = np.cumsum([quad(lambda t: omega_gw(t), t[i-1], t[i], limit=500, epsrel=acc, epsabs=acc)[0] if not i == 0 else 0. for i in range(len(t)) ])

    # and the derivative of omega_gw
    domega_gw= omega_gw.derivative()

    # Calculate PhiTilde
    Phi = Phi - Phi[-1]
    t_c= (t[-1] + 5./256. * R[-1]**4/sp.m_total()**2 / sp.m_reduced())
    tpt = 2.*np.pi*f_gw* (t- t_c)
    PhiTild = tpt - Phi

    # Now compute the time-dependant amplitude A
    A = 1./sp.D * 4. *sp.redshifted_m_reduced() * omega_s**2 * R**2

    # The phase of the GW signal is given by the steady state aproximation
    Psi = 2.*np.pi*f_gw*sp.D + PhiTild - np.pi/4.

    # This gives us h on the f grid (accounting for redshift)
    h_plus =  1./2. *  A * np.sqrt(2*np.pi * (1+sp.z())**2 / domega_gw(t_of_f(f_gw))) * (1. + np.cos(sp.inclination_angle)**2)/2.
    h_cross =  1./2. *  A * np.sqrt(2*np.pi * (1+sp.z())**2 / domega_gw(t_of_f(f_gw))) * np.cos(sp.inclination_angle)

    if dbg:
        return f_gw/(1.+sp.z()), h_plus, h_cross, Psi, t_of_f, PhiTild, A

    return f_gw/(1.+sp.z()), h_plus, h_cross, Psi


def h_n(n, sp, t, a, e, acc=1e-13):
    """
    This function calculates the gravitational waveform h^n_+ for eccentric inspirals according to eq (101) in https://arxiv.org/pdf/2107.00741.pdf
    Parameters:
        sp (SystemProp) : The object describing the properties of the inspiralling system
        t  (array_like) : The time steps of the system evolution
        a  (array_like) : The corresponding semi-major axes at the time steps
        e  (array_like) : The corresponding eccentricities at the time steps
        acc    (float)  : An accuracy parameter that is passed to the integration function

    Returns:
    return f_gw, h_n_plus, h_n_cross, Psi_n
        f_gw : np.ndarray
            The frequencies of the gravitational wave emission at the corresponding time steps
        h_n_plus : np.ndarray
            The amplitude of the plus polarization waveform in the fourier domain of the nth harmonic at the corresponding time steps
        h_n_cross : np.ndarray
            The amplitude of the cross polarization waveform in the fourier domain of the nth harmonic at the corresponding time steps
        Psi_n : np.ndarry
            The phase of the waveform in the fourier domain of the nth harmonic at the corresponding time steps

    TODO:
        Check redshift inclusion
    """

    def C_n_plus(n, sp, e):
        return - (  2.* np.sin(sp.inclination_angle)**2 * jv(n, n*e)
                    -  2./e**2 * (1. + np.cos(sp.inclination_angle)**2) * np.cos(2.*sp.pericenter_angle)
                              * ((e**2 - 2.)*jv(n, n*e) + n*e*(1.-e**2) * (jv(n-1, n*e) - jv(n+1, n*e))) )

    def S_n_plus(n, sp, e):
        return - ( 2./e**2 * np.sqrt(1. - e**2) * (1. + np.cos(sp.inclination_angle)**2) * np.sin(2.*sp.pericenter_angle)
                            * ( -2.*(1.-e**2)*n*jv(n, n*e) + e*(jv(n-1, n*e) - jv(n+1, n*e)) ) )

    def C_n_cross(n, sp, e):
        return - ( 4./e**2 * np.cos(sp.inclination_angle)*np.sin(2.*sp.pericenter_angle)
                          * ( (e**2 - 2.)*jv(n, n*e) +  n*e*(1.-e**2)*(jv(n-1, n*e) - jv(n+1, n*e)) ) )

    def S_n_cross(n, sp, e):
        return - ( 4./e**2 * np.sqrt(1. - e**2) * np.cos(sp.inclination_angle) * np.cos(2.*sp.pericenter_angle)
                            * ( -2.*(1.-e**2)*n*jv(n, n*e) +  e*(jv(n-1, n*e) - jv(n+1, n*e)) ) )

    # Calculate the Keplerian orbital frequency and its derivative over time
    F = np.sqrt(sp.m_total()/a**3) / 2./np.pi
    F_dot = np.gradient(F, t)

    # Calculate the mean anomaly of the orbit
    F_interp = interp1d(t, F, kind='cubic', bounds_error=True)
    mean_anomaly = np.cumsum([quad(F_interp, t[i-1], t[i], epsabs=acc, epsrel=acc, limit=100)[0] if i > 0 else 0. for i in range(len(t))])

    # calculate coalescense time left at the end of the a,e data
    t_coal =  5./256. * a[-1]**4/sp.m_total()**2 /sp.m_reduced()
    def g(e):
        return e**(12./19.)/(1. - e**2) * (1. + 121./304. * e**2)**(870./2299.)
    t_coal = t_coal * 48./19. / g(e[-1])**4 * quad(lambda e: g(e)**4 *(1-e**2)**(5./2.) /e/(1. + 121./304. * e**2), 0., e[-1], limit=100)[0]   # The inspiral time according to Maggiore (2007)

    # Now we can calculate the phase of the stationary phase approximation
    Psi_n = + 2.*np.pi * F/n * (t - t_coal) - n*mean_anomaly - np.pi/4.

    # Amplitude of the signal
    A = - sp.redshifted_m_chirp()**(5./3.) / sp.D / 2. * (2.*np.pi * F/(1.+sp.z()))**(2./3.) / np.sqrt(n*F_dot/(1.+sp.z())**2)

    # the actual waveform
    h_n_plus  = A * ( C_n_plus(n, sp, e)   +  1.j  * S_n_plus(n, sp, e))
    h_n_cross = A * ( C_n_cross(n, sp, e)  +  1.j  * S_n_cross(n, sp, e))

    # the corresponding observed frequencies
    f_gw = n*F / (1.+sp.z())

    return f_gw, h_n_plus, h_n_cross, Psi_n



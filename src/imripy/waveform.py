import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad, solve_ivp
import imripy.merger_system as ms
from scipy.special import jv


def h_2(sp, ev, dbg=False, acc=1e-13):
    """
    This function calculates the gravitational waveform h_+,x according to eq (25) in https://arxiv.org/pdf/1408.3534.pdf

    Parameters:
        sp (merger_system.SystemProp)   : The object describing the properties of the inspiralling system
        ev (inspiral.Classic.Evolution) : The evolution object that results from the inspiral modeling
        dbg (bool)      : A parameter returning intermediate variables
        acc    (float)  : An accuracy parameter that is passed to the integration function

    Returns:
    For dbg = False
        f_gw : np.ndarray
            The frequencies of the gravitational wave emission at the corresponding time steps
        h_n_plus : np.ndarray
            The amplitude of the plus polarization waveform in the fourier domain at the corresponding time steps
        h_n_cross : np.ndarray
            The amplitude of the cross polarization waveform in the fourier domain at the corresponding time steps
        Psi : np.ndarray
            The phase of the waveform in the fourier domain at the corresponding time steps

    For dbg = True
        t_of_f : scipy.interp1d
            The interpolation object that maps time and frequency in the inspiral frame
        PhiTild : np.ndarray
            The phase that is left to observe at a given time step
        A   : np.ndarray
            The amplitude of the waveform over time
    """
    # First, obtain mapping of gw frequency and time
    omega_s = sp.omega_s(ev.R)
    f_gw = omega_s / np.pi
    t_of_f = interp1d(f_gw, ev.t, kind='cubic', bounds_error=False, fill_value='extrapolate')

    # Next, get the accumulated phase Phi
    omega_gw = 2.*omega_s
    omega_gw_interp = interp1d(ev.t, omega_gw, kind='cubic', bounds_error=True)
    Phi = np.cumsum([quad(lambda t: omega_gw_interp(t), ev.t[i-1], ev.t[i], limit=200, epsrel=acc, epsabs=acc)[0] if not i == 0 else 0. for i in range(len(ev.t)) ])

    # and the derivative of omega_gw
    domega_gw = np.gradient(omega_gw, ev.t)

    # Calculate PhiTilde
    Phi = Phi - Phi[-1]
    t_c= (ev.t[-1] + 5./256. * ev.R[-1]**4/sp.m_total()**2 / sp.m_reduced())
    tpt = 2.*np.pi*f_gw* (ev.t- t_c)
    PhiTild = tpt - Phi

    # Now compute the time-dependant amplitude A
    A = 1./sp.D * 4. *sp.redshifted_m_reduced() * omega_s**2 * ev.R**2

    # The phase of the GW signal is given by the steady state aproximation
    Psi = 2.*np.pi*f_gw*sp.D + PhiTild - np.pi/4.

    # This gives us h on the f grid (accounting for redshift)
    h_plus =  1./2. *  A * np.sqrt(2*np.pi * (1+sp.z())**2 / domega_gw) * (1. + np.cos(sp.inclination_angle)**2)/2.
    h_cross =  1./2. *  A * np.sqrt(2*np.pi * (1+sp.z())**2 / domega_gw) * np.cos(sp.inclination_angle)

    if dbg:
        return f_gw/(1.+sp.z()), h_plus, h_cross, Psi, t_of_f, PhiTild, A

    return f_gw/(1.+sp.z()), h_plus, h_cross, Psi


def h_n(n, sp, ev, dbg=False, acc=1e-13):
    """
    This function calculates the gravitational waveform h^n_+ for eccentric inspirals according to eq (101) in https://arxiv.org/pdf/2107.00741.pdf
    Parameters:
        sp (merger_system.SystemProp)   : The object describing the properties of the inspiralling system
        ev (inspiral.Classic.Evolution) : The evolution object that results from the inspiral modeling
        dbg (bool)      : A parameter returning intermediate variables
        acc    (float)  : An accuracy parameter that is passed to the integration function

    Returns:
        f_gw : np.ndarray
            The frequencies of the gravitational wave emission at the corresponding time steps
        h_n_plus : np.ndarray
            The amplitude of the plus polarization waveform in the fourier domain of the nth harmonic at the corresponding time steps
        h_n_cross : np.ndarray
            The amplitude of the cross polarization waveform in the fourier domain of the nth harmonic at the corresponding time steps
        Psi_n : np.ndarry
            The phase of the waveform in the fourier domain of the nth harmonic at the corresponding time steps

    For dbg = True
        PhiTild_n : np.ndarray
            The phase that is left to observe at a given time step
        A_n   : np.ndarray
            The amplitude of the waveform over time
    TODO:
        Check redshift, luminosity distance inclusion
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
    F = np.sqrt(sp.m_total(ev.a)/ev.a**3) / 2./np.pi
    F_dot = np.gradient(F, ev.t)

    # Calculate the mean anomaly of the orbit
    F_interp = interp1d(ev.t, F, kind='cubic', bounds_error=True)
    mean_anomaly = 2.*np.pi*  np.cumsum([quad(F_interp, ev.t[i-1], ev.t[i], epsabs=acc, epsrel=acc, limit=200)[0] if i > 0 else 0. for i in range(len(ev.t))])

    # calculate coalescense time left at the end of the a,e data
    t_coal =  5./256. * ev.a[-1]**4/sp.m_total()**2 /sp.m_reduced()    # The circular case
    def g(e):
        return e**(12./19.)/(1. - e**2) * (1. + 121./304. * e**2)**(870./2299.)
    t_coal = t_coal * 48./19. / g(ev.e[-1])**4 * quad(lambda e: g(e)**4 *(1-e**2)**(5./2.) /e/(1. + 121./304. * e**2), 0., ev.e[-1], limit=100)[0]   # The eccentric inspiral time according to Maggiore (2007)
    t_coal = ev.t[-1] + t_coal

    # Now we can calculate the phase of the stationary phase approximation
    PhiTild_n =  2.*np.pi * n * F * (ev.t - t_coal) - n * (mean_anomaly - mean_anomaly[-1])
    Psi_n = PhiTild_n - np.pi/4.   # TODO: Check inclusion of D term

    # Amplitude of the signal
    A_n = - sp.redshifted_m_chirp()**(5./3.) / sp.D / 2. * (2.*np.pi * F/(1.+sp.z()))**(2./3.) / np.sqrt(n*F_dot/(1.+sp.z())**2) # TODO: Check redshift factors
    A_n = np.where(F_dot == 0., 0., A_n)

    # the actual waveform
    h_n_plus  = A_n * ( C_n_plus(n, sp, ev.e)   +  1.j  * S_n_plus(n, sp, ev.e))
    h_n_cross = A_n * ( C_n_cross(n, sp, ev.e)  +  1.j  * S_n_cross(n, sp, ev.e))

    # the corresponding observed frequencies
    f_gw = n*F / (1.+sp.z())

    if dbg:
        return f_gw, h_n_plus, h_n_cross, Psi_n, PhiTild_n, A_n

    return f_gw, h_n_plus, h_n_cross, Psi_n



def h(sp, ev, t_grid, phi_0=0., acc=1e-13):
    """
    This function calculates the time domain gravitational waveform h_+,x(t) for eccentric inspirals according to eq (96) in https://arxiv.org/pdf/2107.00741.pdf
    Parameters:
        sp (merger_system.SystemProp)   : The object describing the properties of the inspiralling system
        ev (inspiral.Classic.Evolution) : The evolution object that results from the inspiral modeling
        t_grid (array_like) : The times at which to evaluate h
        phi_0 (float)   : The initial phase of the orbit at t_grid[0]
        acc    (float)  : An accuracy parameter that is passed to the integration function

    Returns:
        h_plus : np.ndarray
            The amplitude of the plus polarization waveform at the corresponding time steps of t_grid
        h_cross : np.ndarray
            The amplitude of the cross polarization waveform at the corresponding time steps of t_grid
    """
    a_int = interp1d(ev.t, ev.a, kind='cubic', bounds_error=True)
    e_int = interp1d(ev.t, ev.e, kind='cubic', bounds_error=True)

    def phi_dot(t, phi):  # The orbital phase evolution according to Maggiore (2007)
        return np.sqrt(sp.m_total()/a_int(t)**3) * (1. - e_int(t)**2)**(-3./2.) * (1. + e_int(t)*np.cos(phi))**2

    sol = solve_ivp(phi_dot, [t_grid[0], t_grid[-1]], [phi_0], t_eval=t_grid, rtol=acc, atol=acc)  # calculate the orbital phase at the given time steps t_grid
    phi = sol.y[0]

    h_plus = - ( (2. * np.cos(2.*phi - 2.*sp.pericenter_angle) + 5.*e_int(t_grid)/2. * np.cos(phi - 2.*sp.pericenter_angle)
                    + e_int(t_grid)/2. * np.cos(3.*phi - 2.*sp.pericenter_angle) + e_int(t_grid)**2 * np.cos(2.*sp.pericenter_angle) ) * (1. + np.cos(sp.inclination_angle)**2 )
            + (e_int(t_grid) * np.cos(phi) + e_int(t_grid)**2 ) * np.sin(sp.inclination_angle)**2 )
    h_plus *= sp.m_reduced() / (a_int(t_grid)*(1. - e_int(t_grid)**2)) / sp.D

    h_cross = - (4. * np.sin(2.*phi - 2.*sp.pericenter_angle) + 5.*e_int(t_grid) * np.sin(phi - 2.*sp.pericenter_angle)
                        + e_int(t_grid) * np.sin(3.*phi - 2.*sp.pericenter_angle) - 2.*e_int(t_grid)**2 * np.sin(2.*sp.pericenter_angle)) * np.cos(sp.inclination_angle)
    h_cross *= sp.m_reduced() / (a_int(t_grid)*(1. - e_int(t_grid)**2)) / sp.D

    return h_plus, h_cross


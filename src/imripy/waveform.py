import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad, solve_ivp, cumulative_trapezoid
from scipy.special import jv
import collections
from imripy.kepler import KeplerOrbit


def h_2(hs, ev, dbg=False, acc=1e-13):
    """
    This function calculates the gravitational waveform h_+,x according to eq (25) in https://arxiv.org/pdf/1408.3534.pdf

    Parameters:
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
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
    omega_s = hs.omega_s(ev.R)
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
    t_c= (ev.t[-1] + 5./256. * ev.R[-1]**4/ev.m_tot**2 / ev.m_red)
    tpt = 2.*np.pi*f_gw* (ev.t- t_c)
    PhiTild = tpt - Phi

    # Now compute the time-dependant amplitude A
    A = 1./hs.D_l * 4. *ev.redshifted_m_red * omega_s**2 * ev.R**2

    # The phase of the GW signal is given by the steady state aproximation
    Psi = 2.*np.pi*f_gw*hs.D_l + PhiTild - np.pi/4.

    # This gives us h on the f grid (accounting for redshift)
    h_plus =  1./2. *  A * np.sqrt(2*np.pi * (1+hs.z)**2 / domega_gw) * (1. + np.cos(ev.inclination_angle)**2)/2. # TODO: This is only in the fundamental frame
    h_cross =  1./2. *  A * np.sqrt(2*np.pi * (1+hs.z)**2 / domega_gw) * np.cos(ev.inclination_angle)

    if dbg:
        return f_gw/(1.+hs.z), h_plus, h_cross, Psi, t_of_f, PhiTild, A

    return f_gw/(1.+hs.z), h_plus, h_cross, Psi


def h_n(n, hs, ev, dbg=False, acc=1e-13):
    """
    This function calculates the gravitational waveform h^n_+ for eccentric inspirals according to eq (101) in https://arxiv.org/pdf/2107.00741.pdf
    Parameters:
        n (int) : The harmonic of interest, must be a positive integer
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
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
    inclination_angle, beta = get_observer_orbit_angles(hs, ev, ev.t)
    s_i = np.sin(inclination_angle); c_i = np.cos(inclination_angle)

    def C_n_plus(n, periapse_angle, e):
        return  (  2.* s_i**2 * jv(n, n*e) +  2./e**2 * (1. + c_i**2) * np.cos(2.*periapse_angle)
                              * ((e**2 - 2.)*jv(n, n*e) + n*e*(1.-e**2) * (jv(n-1, n*e) - jv(n+1, n*e))) )

    def S_n_plus(n, periapse_angle, e):
        return - ( 2./e**2 * np.sqrt(1. - e**2) * (1. + c_i**2) * np.sin(2.*periapse_angle)
                            * ( -2.*(1.-e**2)*n*jv(n, n*e) + e*(jv(n-1, n*e) - jv(n+1, n*e)) ) )

    def C_n_cross(n, periapse_angle, e):
        return - ( 4./e**2 * c_i*np.sin(2.*periapse_angle)
                          * ( (2. - e**2)*jv(n, n*e) +  n*e*(1.-e**2)*(jv(n-1, n*e) - jv(n+1, n*e)) ) )

    def S_n_cross(n, periapse_angle, e):
        return - ( 4./e**2 * np.sqrt(1. - e**2) * c_i * np.cos(2.*periapse_angle)
                            * ( -2.*(1.-e**2)*n*jv(n, n*e) +  e*(jv(n-1, n*e) - jv(n+1, n*e)) ) )

    # Calculate the Keplerian orbital frequency and its derivative over time
    F = np.sqrt(ev.m_tot/ev.a**3) / 2./np.pi
    F_dot = np.gradient(F, ev.t)

    # Calculate the mean anomaly of the orbit
    F_interp = interp1d(ev.t, F, kind='cubic', bounds_error=True)
    mean_anomaly = 2.*np.pi*  np.cumsum([quad(F_interp, ev.t[i-1], ev.t[i], epsabs=acc, epsrel=acc, limit=200)[0] if i > 0 else 0. for i in range(len(ev.t))])

    # calculate coalescense time left at the end of the a,e data
    t_coal =  5./256. * ev.a[-1]**4/ev.m_tot**2 / ev.m_red    # The circular case
    def g(e):
        return e**(12./19.)/(1. - e**2) * (1. + 121./304. * e**2)**(870./2299.)
    if ev.e[-1] > 0.:
        t_coal = t_coal * 48./19. / g(ev.e[-1])**4 * quad(lambda e: g(e)**4 *(1-e**2)**(5./2.) /e/(1. + 121./304. * e**2), 0., np.clip(ev.e[-1], 0.,1.), limit=100)[0]   # The eccentric inspiral time according to Maggiore (2007)
    t_coal = ev.t[-1] + t_coal

    # Now we can calculate the phase of the stationary phase approximation
    PhiTild_n =  2.*np.pi * n * F * (ev.t - t_coal) - n * (mean_anomaly - mean_anomaly[-1])
    Psi_n = PhiTild_n - np.pi/4.   # TODO: Check inclusion of D term

    # Amplitude of the signal
    A_n = - ev.redshifted_m_chirp**(5./3.) / hs.D_l / 2. * (2.*np.pi * F/(1.+hs.z))**(2./3.) / np.sqrt(n*F_dot/(1.+hs.z)**2) # TODO: Check redshift factors
    A_n = np.where(F_dot == 0., 0., A_n)

    # the actual waveform
    e = np.clip(ev.e, 1e-10, None) # in case eccentricity vanishes
    h_n_plus  = A_n * ( C_n_plus(n, beta, e)   +  1.j  * S_n_plus(n, beta, e))
    h_n_cross = A_n * ( C_n_cross(n, beta, e)  +  1.j  * S_n_cross(n, beta, e))

    # the corresponding observed frequencies
    f_gw = n*F / (1.+hs.z)

    if dbg:
        return f_gw, h_n_plus, h_n_cross, Psi_n, PhiTild_n, A_n

    return f_gw, h_n_plus, h_n_cross, Psi_n


def get_observer_orbit_angles(hs, ev, t_grid):
    """
    This funtion calculates the angles between the orbital and observer plane,
        by going through the fundamental frame
    The resulting angles beta', iota' describe the angle between the polarization axis in the observer plane and the periapse in the orbital plane,
        and the inclination of the orbital plane wrt the observer plane.
    For the math see dissertation.

    Parameters:
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
        ev (inspiral.Classic.Evolution) : The evolution object that results from the inspiral modeling
        t_grid (array_like) : The times at which to evaluate the angles, will call the _int interpolation objects in ev

    Returns:
        iota_prime : np.ndarray
            The inclination angle of the observer plane wrt the orbital plane
        beta_prime : np.ndarray
            The angle between the polarization axis in the observer frame to the periapse of the orbit
    """
    omega = ev.periapse_angle_int(t_grid)
    Omega = ev.longitude_an_int(t_grid)
    iota = ev.inclination_angle_int(t_grid)
    iota_tilde = hs.inclination_angle

    iota_prime = np.arccos(np.cos(iota)*np.cos(iota_tilde)
                                    - np.sin(iota)*np.sin(iota_tilde)*np.cos(Omega)
                                 )

    if np.all(iota == 0.) or iota_tilde == 0.:
        beta = (omega + Omega)

    elif np.all(Omega == 0.):
        beta = omega

    elif np.all(omega == 0.):
        omega_tilde = np.arctan2(np.sin(Omega)*np.sin(iota_tilde),
                                 np.cos(Omega) *np.cos(iota)*np.sin(iota_tilde)
                                  + np.sin(iota)*np.cos(iota_tilde))

        Omega_tilde = np.arctan2( np.sin(Omega)*np.sin(iota),
                                   np.sin(iota)*np.cos(Omega)*np.cos(iota_tilde)
                                     + np.sin(iota_tilde)*np.cos(iota))
        beta = (omega_tilde + Omega_tilde)

    else:
        omega_tilde = -np.arctan2( ( np.cos(Omega) * np.sin(omega) * np.cos(iota)
                                        + np.sin(Omega) * np.cos(omega) ) * np.sin(iota_tilde)
                                     + np.sin(iota) * np.sin(omega) * np.cos(iota_tilde),

                                     ( -np.cos(Omega) * np.cos(omega) * np.cos(iota)
                                       + np.sin(Omega) * np.sin(omega) ) * np.sin(iota_tilde)
                                     - np.sin(iota) * np.cos(omega) * np.cos(iota_tilde)
                               )
        Omega_tilde = np.arctan2( np.sin(Omega)*np.sin(iota),
                                   np.sin(iota)*np.cos(Omega)*np.cos(iota_tilde)
                                     + np.sin(iota_tilde)*np.cos(iota))

        beta = (omega_tilde + Omega_tilde)
    return iota_prime, beta


def h(hs, ev, t_grid, phi_0=0., acc=1e-13):
    """
    This function calculates the time domain gravitational waveform h_+,x(t) for eccentric inspirals according to eq (96) in https://arxiv.org/pdf/2107.00741.pdf
    Should be identical to h_projected, up to numerical inaccuracies.

    Parameters:
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
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

    def phi_dot(t, phi):  # The orbital phase evolution according to Maggiore (2007)
        return np.sqrt(ev.m_tot/ev.a_int(t)**3) * (1. - ev.e_int(t)**2)**(-3./2.) * (1. + ev.e_int(t)*np.cos(phi))**2

    sol = solve_ivp(phi_dot, [t_grid[0], t_grid[-1]], [phi_0], t_eval=t_grid, rtol=acc, atol=acc)  # calculate the orbital phase at the given time steps t_grid
    phi = sol.y[0]
    e = ev.e_int(t_grid)

    inclination_angle, beta = get_observer_orbit_angles(hs, ev, t_grid)

    h_plus = - ( (2. * np.cos(2.*phi + 2.*beta) + 5.*e/2. * np.cos(phi + 2.*beta)
                    + e/2. * np.cos(3.*phi + 2.*beta) + e**2 * np.cos(2.*beta) ) * (1. + np.cos(inclination_angle)**2 )
                + (e * np.cos(phi) + e**2 ) * np.sin(inclination_angle)**2 )
    h_plus *= ev.m_red*ev.m_tot / (ev.a_int(t_grid)*(1. - e**2)) / hs.D_l

    h_cross = - (4. * np.sin(2.*phi + 2.*beta) + 5.*e * np.sin(phi + 2.*beta)
                        + e * np.sin(3.*phi + 2.*beta) + 2.*e**2 * np.sin(2.*beta)) * np.cos(inclination_angle)
    h_cross *= ev.m_red*ev.m_tot / (ev.a_int(t_grid)*(1. - e**2)) / hs.D_l

    return h_plus, h_cross

def h_projected(hs, ev, t_grid, phi_0=0., acc=1e-13):
    """
    This function calculates the time domain gravitational waveform h_+,x(t) for eccentric inspirals
        according to 11.2.2 in Poisson, Will (2014)
        by calculating the emissions in the fundamental frame and projecting it onto the observer plane.
    Should be identical with h(), up to numerical inaccuracies

    Parameters:
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
        ev (inspiral.Classic.Evolution) : The evolution object that results from the inspiral modeling
        t_grid (array_like) : The times at which to evaluate h
        phi_0 (float)   : The initial phase of the orbit at t_grid[0]
        acc    (float)  : An accuracy parameter that is passed to the integration function

    Returns:
        h_plus : np.ndarray
            The amplitude of the plus polarization waveform at the corresponding time steps of t_grid
        h_cross : np.ndarray
            The amplitude of the cross polarization waveform at the corresponding time steps of t_grid

    Comments:
        We have three frames here: The orbital plane of the inspiral, the fundamental frame of the hs system,
        and the observers frame. The angles in the ev object give the orbital plane wrt to the fundamental frame,
        and the angles in the hs system give the fundamental frame wrt to the observers frame.
        The waveforms can be described in the orbital plane and then rotated into the two other frames consecutively
    """
    # First, we have to get the phase of the secondary on t_grid
    def phi_dot(t, phi):  # The orbital phase evolution according to Maggiore (2007)
        return np.sqrt(ev.m_tot/ev.a_int(t)**3) * (1. - ev.e_int(t)**2)**(-3./2.) * (1. + ev.e_int(t)*np.cos(phi))**2

    sol = solve_ivp(phi_dot, [t_grid[0], t_grid[-1]], [phi_0], t_eval=t_grid, rtol=acc, atol=acc)  # calculate the orbital phase at the given time steps t_grid
    phi_grid = sol.y[0]

    # Go from the fundamental frame to the observers frame to obtain polarization axis
    ko_observer = KeplerOrbit(None, None, None, inclination_angle=hs.inclination_angle)
    X_obs = ko_observer.from_fundamental_xy_plane_to_orbital_xy_plane(np.array([1.,0.,0.]))
    Y_obs = ko_observer.from_fundamental_xy_plane_to_orbital_xy_plane(np.array([0.,1.,0.]))

    # Calculate the waveforms
    h_plus, h_cross = np.zeros(np.shape(t_grid)), np.zeros(np.shape(t_grid))
    for i, (t, phi) in enumerate(zip(t_grid, phi_grid)):
        ko = ev.get_kepler_orbit(t, interpolate=True)
        n, m, _ = ko.get_orbital_decomposition_in_fundamental_xy_plane(phi)

        h_jk = (- (1. + ko.e* np.cos(phi) - ko.e**2 * np.sin(phi)**2) * np.outer(n,n)
                + ko.e*np.sin(phi) * (1.+ ko.e*np.cos(phi))*(np.outer(n,m) + np.outer(m,n))
                + (1.+ko.e*np.cos(phi))**2 * np.outer(m, m) )

        # Project to get polarizations
        h_plus[i] = np.matmul(X_obs.T, np.matmul(h_jk, X_obs)) -  np.matmul(Y_obs.T, np.matmul(h_jk, Y_obs))
        h_cross[i] = np.matmul(X_obs.T, np.matmul(h_jk, Y_obs)) +  np.matmul(Y_obs.T, np.matmul(h_jk, X_obs))

    # Multiply to get the right units
    h_plus *= 2.*ev.m_red*ev.m_tot / (ev.a_int(t_grid)*(1. - ev.e_int(t_grid)**2)) / hs.D_l
    h_cross *= 2.*ev.m_red*ev.m_tot / (ev.a_int(t_grid)*(1. - ev.e_int(t_grid)**2)) / hs.D_l

    return h_plus, h_cross

def N_cycles_n(n, hs, ev, acc=1e-13):
    """
    Calculates the amount of cycles of a given harmonic n left to observe for a given frequency as given by eq. (5.3) of https://arxiv.org/pdf/2002.12811.pdf with t_f = ev.t[-1]

    Parameters:
        n (int) : The harmonic of interest, must be a positive integer
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
        ev (inspiral.Classic.Evolution) : The evolution object that results from the inspiral modeling
        dbg (bool)      : A parameter returning intermediate variables
        acc    (float)  : An accuracy parameter that is passed to the integration function

    Returns:
        f_gw : np.ndarray
            The frequencies of the harmonic
        N_cycles : np.ndarray
            The number of cycles left to observe for that harmonic
    """
    F = np.sqrt(ev.m_tot/ev.a**3) / 2./np.pi
    N = cumulative_trapezoid(F[::-1], ev.t[::-1], initial=0)[::-1]
    N *= -n
    return n*F, N


def BrakingIndex(hs, ev, acc=1e-13):
    """
    Calculates the braking index as originally defined by the spindown of neutron stars (see eq (4) of https://arxiv.org/pdf/2209.10981.pdf)
        as f * f'' / (f')^2
        where ' denotes the temporal derivative

    Parameters:
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
        ev (inspiral.Classic.Evolution) : The evolution object that results from the inspiral modeling
        acc    (float)  : An accuracy parameter that is passed to the integration function

    Returns:
        F : np.ndarray
            The frequencies of the lowest harmonic
        n : np.ndarray
            The braking index
    """
    F = np.sqrt(ev.m_tot/ev.a**3) / 2./np.pi
    dF = np.gradient(F, ev.t)
    ddF = np.gradient(dF, ev.t)
    return F, F*ddF/ dF**2

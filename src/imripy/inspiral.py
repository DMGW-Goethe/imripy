import numpy as np
from scipy.integrate import solve_ivp, quad, simps
from scipy.interpolate import griddata
from scipy.special import ellipeinc, ellipe, ellipkinc, factorial, factorial2, hyp2f1
import collections
#import sys
import time
import imripy.merger_system as ms



class Classic:
    """
    A class bundling the functions to simulate an inspiral with basic energy conservation arguments
    This class does not need to be instantiated

    Attributes:
        ln_Lambda (float): The Coulomb logarithm of the dynamical friction description. Set -1 for ln sqrt(m1/m2). Default is 3.
        dmPhaseSpaceFraction (float) : As the dm particles in the halo are not stationary, the relative velocity effects need to be modeled, here according to https://arxiv.org/pdf/2002.12811.pdf
    """
    ln_Lambda = 3.
    dmPhaseSpaceFraction = 0.58


    def E_orbit(sp, a, e=0.):
        """
        The function gives the orbital energy of the binary with central mass m1 with the surrounding halo and the smaller mass m2
           for a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit, default is 0 - a circular orbit

        Returns:
            out : float
                The energy of the Keplerian orbit
        """
        return  - sp.m_total(a)*sp.m_reduced(a) / a / 2.

    def dE_orbit_da(sp, a, e=0.):
        """
        The function gives the derivative of the orbital energy wrt the semimajor axis a
           of the binary with central mass m1 with the surrounding halo and the smaller mass m2
           for a Keplerian orbit with semimajor axis a and eccentricity e
        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
        Returns:
            out : float
                The derivative of the orbital energy wrt to a of the Keplerian orbit
        """
        return sp.m2 * sp.mass(a) / 2. / a**2  * ( 1.
                                                        - a*sp.dmass_dr(a)/sp.mass(a)
                                                    )

    def L_orbit(sp, a, e):
        """
        The function gives the angular momentum of the binary with central mass m1 with the surrounding halo and the smaller mass m2
           for a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit

        Returns:
            out : float
                The angular momentum of the Keplerian orbit
        """
        return np.sqrt( -(1. - e**2) * sp.m_reduced(a)**3 * sp.m_total(a)**2 / 2. / Classic.E_orbit(sp, a, e))


    def dE_orbit_da(sp, a, e=0.):
        """
        The function gives the derivative of the orbital energy wrt the semimajor axis a
           of the binary with central mass m1 with the surrounding halo and the smaller mass m2
           for a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit

        Returns:
            out : float
                The derivative of the orbital energy wrt to a of the Keplerian orbit
        """
        return sp.m2 * sp.mass(a) / 2. / a**2  * ( 1.
                                                     - 4.*np.pi * sp.halo.density(a)* a**3 / sp.mass(a)
                                                     )


    def dE_gw_dt(sp, a, e=0.):
        """
        The function gives the energy loss due to radiation of gravitational waves
            for a Keplerian orbit with semimajor axis a and eccentricity e
        According to Maggiore (2007)

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit

        Returns:
            out : float
                The energy loss due to radiation of gravitational waves of an Keplerian orbit
        """
        return -32./5. * sp.m_reduced(a)**2 * sp.m_total(a)**3 / a**5  / (1. - e**2)**(7./2.) * (1. + 73./24. * e**2 + 37./96. * e**4)


    def F_df(sp, r, v):
        """
        The function gives the force of the dynamical friction of an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v
        The ln_Lambda is the Coulomb logarithm, for which different authors use different values. Set to -1 so that Lambda = sqrt(m1/m2)

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float)      : The speed of the orbiting object wrt to the dark matter halo

        Returns:
            out : float
                The magnitude of the dynamical friction force
        """
        ln_Lambda = Classic.ln_Lambda
        if ln_Lambda < 0.:
            ln_Lambda = np.log(sp.m1/sp.m2)/2.
        return 4.*np.pi * sp.m2**2 * sp.halo.density(r) * Classic.dmPhaseSpaceFraction * ln_Lambda / v**2


    def dE_df_dt(sp, a, e=0.):
        """
        The function gives the energy loss due to dynamical friction of the smaller object with the dark matter halo
           on a Keplerian orbit with semimajor axis a and eccentricity e
        For a circular orbit the Chandrasekhar equation is used,
            for an elliptic orbit the expression of https://arxiv.org/abs/1908.10241 is used

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit

        Returns:
            out : float
                The energy loss due to dynamical friction
        """
        if e == 0.:
            v_s = sp.omega_s(a)*a
            return - Classic.F_df(sp, a, v_s)*v_s
        else:
            if  isinstance(a, (collections.Sequence, np.ndarray)):
                return np.array([Classic.dE_df_dt(sp, a_i, e) for a_i in a])
            def integrand(phi):
                r = a*(1. - e**2)/(1. + e*np.cos(phi))
                v_s = np.sqrt(sp.m_total(a) *(2./r - 1./a))
                return Classic.F_df(sp, r, v_s)*v_s / (1.+e*np.cos(phi))**2
            return -(1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]

    def BH_cross_section(sp, v):
        """
        The function gives the cross section of a small black hole (m2) moving through a halo of particles
            according to https://arxiv.org/pdf/1711.09706.pdf

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system, the small black hole is taken to be sp.m2
            v  (float)      : The relative velocity of the black hole to the halo

        Returns:
            out : float
                The black hole cross section
        """
        #return 16. * np.pi * sp.m2**2 / v**2  * (1. + v**2)
        return (np.pi * sp.m2**2. / v**2.) * (8. * (1. - v**2.))**3 / (4. * (1. - 4. * v**2. + (1. + 8. * v**2.)**(1./2.)) * (3. - (1. + 8. * v**2.)**(1./2.))**2.)


    def dm2_dt(sp, r, v):
        """
        The function gives the mass gain due to accretion of the small black hole inside of the dark matter halo
           for a small black hole with relative velocity v to the halo at radius r
        The equation of https://arxiv.org/pdf/1711.09706.pdf is used

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radial position of the black hole in the halo
            v  (float)      : The relative velocity

        Returns:
            out : float
                The mass gain due to accretion
        """
        return sp.halo.density(r) * v * Classic.BH_cross_section(sp, v)


    def dm2_dt_avg(sp, a, e=0.):
        """
        The function gives the mass gain due to accretion of the small black hole inside of the dark matter halo
           on a Keplerian orbit with semimajor axis a and eccentricity e
        For a circular orbit the dm2_dt function with the corresponding orbital velocity is used
            for an elliptic orbit the average of the expression is used

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit

        Returns:
            out : float
                The mass gain due to accretion on an orbit
        """
        if e == 0.:
            v_s = sp.omega_s(a)*a
            return Classic.dm2_dt(sp, a, v_s)
        else:
            if  isinstance(a, (collections.Sequence, np.ndarray)):
                return np.array([Classic.dm2_dt(sp, a_i, e) for a_i in a])
            def integrand(phi):
                r = a*(1. - e**2)/(1. + e*np.cos(phi))
                v_s = np.sqrt(sp.m_total(a) *(2./r - 1./a))
                return Classic.dm2_dt(sp, r, v_s) / (1.+e*np.cos(phi))**2
            return (1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]

    def F_acc(sp, r, v):
        """
        The function gives the force of the accretion of an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float)      : The speed of the orbiting object wrt to the dark matter halo

        Returns:
            out : float
                The magnitude of the accretion force
        """
        return Classic.dm2_dt(sp, r, v) * v


    def dE_acc_dt(sp, a, e=0.):
        """
        The function gives the energy loss of the orbiting small black hole due to accretion of the dark matter halo
           on a Keplerian orbit with semimajor axis a and eccentricity e
        For a circular orbit the dm2_dt function with the corresponding orbital velocity is used
            for an elliptic orbit the average of the expression is used

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit

        Returns:
            out : float
                The energy loss due to accretion
        """

        if e == 0.:
            v_s = sp.omega_s(a)*a
            return - Classic.F_acc(sp, a, v_s)*v_s
        else:
            if  isinstance(a, (collections.Sequence, np.ndarray)):
                return np.array([Classic.dE_acc_dt(sp, a_i, e) for a_i in a])
            def integrand(phi):
                r = a*(1. - e**2)/(1. + e*np.cos(phi))
                v_s = np.sqrt(sp.m_total(a) *(2./r - 1./a))
                return Classic.F_acc(sp, r, v_s)*v_s / (1.+e*np.cos(phi))**2
            return -(1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    def dE_dt(sp, a, e=0., accretion=False):
        """
        The function gives the total energy loss of the orbiting small black hole due to the dissipative effects
           on a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            accretion (bool): A boolean deciding wether to include accretion effects

        Returns:
            out : float
                The total energy loss
        """
        dE_gw_dt = Classic.dE_gw_dt(sp, a, e)
        dE_df_dt = Classic.dE_df_dt(sp, a, e)
        if accretion:
            dE_acc_dt = Classic.dE_acc_dt(sp, a, e)

        #print("dE_gw_dt=", dE_gw_dt, "dE_df_dt=", dE_df_dt, "dE_acc_dt=", dE_acc_dt if accretion else 0.)
        return ( dE_gw_dt + dE_df_dt + (dE_acc_dt if accretion else 0.))


    def dL_gw_dt(sp, a, e):
        """
        The function gives the loss of angular momentum due to radiation of gravitational waves of the smaller object
           on a Keplerian orbit with semimajor axis a and eccentricity e
        According to Maggiore (2007)

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit

        Returns:
            out : float
                The angular momentum loss due to radiation of gravitational waves
        """
        return -32./5. * sp.m_reduced(a)**2 * sp.m_total(a)**(5./2.) / a**(7./2.)  / (1. - e**2)**2 * (1. + 7./8.*e**2)


    def dL_df_dt(sp, a, e):
        """
        The function gives the angular momentum loss due to dynamical friction of the smaller object with the dark matter halo
           on a Keplerian orbit with semimajor axis a and eccentricity e
        According to https://arxiv.org/abs/1908.10241

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit

        Returns:
            out : float
                The angular momentum loss due to dynamical friction
        """
        def integrand(phi):
            r = a*(1. - e**2)/(1. + e*np.cos(phi))
            v_s = np.sqrt(sp.m_total(a) *(2./r - 1./a))
            return Classic.F_df(sp, r, v_s) / v_s / (1.+e*np.cos(phi))**2
        return -(1.-e**2)**(3./2.)/2./np.pi *np.sqrt(sp.m_total(a)* a*(1.-e**2)) *  quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    def dL_acc_dt(sp, a, e):
        """
        The function gives the angular momentum loss due to accretion of the small black hole inside the dark matter halo
           on a Keplerian orbit with semimajor axis a and eccentricity e
        According to https://arxiv.org/pdf/1711.09706.pdf and https://arxiv.org/abs/1908.10241

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit

        Returns:
            out : float
                The angular momentum loss due to accretion
        """
        def integrand(phi):
            r = a*(1. - e**2)/(1. + e*np.cos(phi))
            v_s = np.sqrt(sp.m_total(a) *(2./r - 1./a))
            return Classic.F_acc(sp, r, v_s) / v_s / (1.+e*np.cos(phi))**2
        return -(1.-e**2)**(3./2.)/2./np.pi *np.sqrt(sp.m_total(a) * a*(1.-e**2)) *  quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    def dL_dt(sp, a, e, accretion=False):
        """
        The function gives the total angular momentum loss of the secondary object
            on a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            accretion (bool): A boolean deciding wether to include accretion effects

        Returns:
            out : float
                The total angular momentum loss
        """
        dL_gw_dt = Classic.dL_gw_dt(sp, a, e)
        dL_df_dt = Classic.dL_df_dt(sp, a, e)
        if accretion:
            dL_acc_dt = Classic.dL_acc_dt(sp, a, e)

        #print("dL_gw_dt=", dL_gw_dt, "dL_df_dt=", dL_df_dt, "dL_acc_dt=", dL_acc_dt if accretion else 0.)
        return  (dL_gw_dt + dL_df_dt + (dL_acc_dt if accretion else 0.))

    def da_dt(sp, a, e=0., accretion=False, dm2_dt = 0.):
        """
        The function gives the secular time derivative of the semimajor axis a (or radius for a circular orbit) due to gravitational wave emission and dynamical friction
            of the smaller object on a Keplerian orbit with semimajor axis a and eccentricity e
        The equation is obtained by the relation
            E = -m_1 * m_2 / 2a

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            accretion (bool): A boolean deciding wether to include accretion effects

        Returns:
            out : float
                The secular time derivative of the semimajor axis
        """
        dE_dt = Classic.dE_dt(sp, a, e, accretion)
        dE_orbit_dm2 = - sp.mass(a)/2./a
        dE_orbit_da = Classic.dE_orbit_da(sp, a, e)

        return  (  ( dE_dt - dE_orbit_dm2 * dm2_dt   )
                    / dE_orbit_da )


    def de_dt(sp, a, e, da_dt, accretion=False, dm2_dt=0.):
        """
        The function gives the secular time derivative of the eccentricity due to gravitational wave emission and dynamical friction
            of the smaller object on a Keplerian orbit with semimajor axis a and eccentricity e
        The equation is obtained by the time derivative of the relation
            e^2 = 1 + 2EL^2 / m_total^2 / m_reduced^3
           as given in Maggiore (2007)

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            accretion(bool) : A parameter wether to include accretion effects of the secondary black hole
            dm2_dt (float)  : If accretion is included, the time derivative of the mass of the secondary black hole

        Returns:
            out : float
                The secular time derivative of the eccentricity
        """
        if e <= 0.:
            return 0.

        g = 1. / sp.m_total(a)**2 / sp.m_reduced(a)**3
        dg_dm2 = -2. / sp.mass(a)**3 / sp.m2**3 * (1. + 3./2. * sp.mass(a) / sp.m2)
        #dg_dm2 = 0.
        dg_da = -2. / sp.mass(a)**3 / sp.m2**3 * (1. + 3./2. * sp.m2 / sp.mass(a)) * sp.dmass_dr(a)
        #dg_da = 0.

        dE_dt = Classic.dE_dt(sp, a, e, accretion)
        E = Classic.E_orbit(sp, a, e)
        dL_dt = Classic.dL_dt(sp, a, e, accretion)
        L = Classic.L_orbit(sp, a, e)

        #print("dE_dt/E=", dE_dt/E, "2dL_dt/L=", 2.*dL_dt/L, "diff=", dE_dt/E + 2.*dL_dt/L, "dg_dt/g=", (dg_dm2*dm2_dt + dg_da*da_dt)/g, )

        return - (1.-e**2)/2./e *(  dE_dt/E
                                    + 2. * dL_dt/L
                                    + (dg_dm2 * dm2_dt + dg_da * da_dt)/g
                                    )


    class Evolution:
        """
        This class keeps track of the evolution of an inspiral.

        Attributes:
            t : np.ndarray
                The time steps of the evolution
            a,R : np.ndarray
                The corresponding values of the semimajor axis - if e=0, this is also called R
            e  : float/np.ndarray
                The corresponding values of the eccentricity, default is zero
            m2 : float/np.ndarray
                The corresponding values of the mass of the secondary object, if accretion is included
            msg : string
                The message of the solve_ivp integration
            sp : merger_system.SystemProp
                The system properties used in the evolution
        """
        def __init__(self, sp, t, a, e=0., m2=None, msg=None):
            self.sp = sp
            self.t = t
            self.a = a
            if not isinstance(e, (collections.Sequence, np.ndarray)) and e == 0.:
                self.R = a
            self.e = e
            if m2 is None:
                self.m2 = sp.m2
            else:
                self.m2 = m2
            self.msg=msg


    def evolve_circular_binary(sp, R_0, R_fin=0., t_0=0., t_fin=None, acc=1e-8, verbose = 1, accretion=False):
        """
        The function evolves the differential equation of the radius of the circular orbit of the inspiralling system
            dR/dt = dE/dt  /  (dE/dR)
            where dE/dt includes the energy loss due to gravitational wave radiation, dynamical friction and possibly accretion

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            R_0  (float)    : The initial orbital radius
            R_fin (float)   : The orbital radius at which to stop evolution
            t_0    (float)  : The initial time
            t_fin  (float)  : The time until the system should be evolved, if None then the estimated coalescence time will be used
            acc    (float)  : An accuracy parameter that is passed to solve_ivp
            verbose (int)   : A parameter describing how verbose the function should be
            accretion(bool) : A parameter wether to include accretion effects of the secondary black hole

        Returns:
            ev : Evolution
                An evolution object that contains the results
        """
        t_coal =  5./256. * R_0**4/sp.m_total()**2 /sp.m_reduced()
        if t_fin is None:
            t_fin = 1.1 * t_coal *( 1. - R_fin**4 / R_0**4)         # This is 10% above the maximum time the system should reach R_fin

        if R_fin == 0.:
            R_fin = sp.r_isco()
        R_scale = R_fin                 # It's nice for the differential solver to rescale the equations
        t_scale = t_fin
        if accretion:
            m_scale = sp.m2

        #t_step_max = t_fin/1e4 / t_coal
        t_step_max = np.inf
        if verbose > 0:
            print("Evolving from ", R_0/sp.r_isco(), " to ", R_fin/sp.r_isco(),"r_isco, " + ("with" if accretion else "without") + " accretion")

        def dy_dt(t, y, *args):
            sp = args[0]
            t = t*t_scale
            R = y[0]*R_scale
            if accretion:
                sp.m2 = y[1]*m_scale

            if verbose > 1:
                tic = time.perf_counter()

            if accretion:
                dm2_dt =  Classic.dm2_dt_avg(sp, R)
            else:
                dm2_dt = 0.
            dR_dt =  Classic.da_dt(sp, R, accretion=accretion, dm2_dt=dm2_dt)

            if verbose > 1:
                toc = time.perf_counter()
                print("t=", t, "R=", R, "dR/dt=", dR_dt, "m2=", sp.m2, "dm2/dt=", dm2_dt,
                                    "elapsed real time: ", toc-tic)

            if accretion:
                return [t_scale/R_scale * dR_dt, t_scale/m_scale * dm2_dt]
            return t_scale/R_scale * dR_dt

        fin_reached = lambda t,y, *args: y[0] - R_fin/R_scale          # Give the termination condition that R = R_fin
        fin_reached.terminal = True

        if accretion:
            y_0 = [R_0/R_scale, sp.m2/m_scale]
        else:
            y_0 = [R_0/R_scale]

        Int = solve_ivp(dy_dt, [t_0/t_scale, (t_0+t_fin)/t_scale], y_0, dense_output=True, args=([sp]), events=fin_reached, max_step=t_step_max/t_scale,
                                                                                        method = 'RK45', atol=acc, rtol=acc)

        R = Int.y[0]*R_scale
        t = Int.t*t_scale
        if accretion:
            m2 = Int.y[1]*m_scale

        if verbose > 0:
            print(Int.message)
            print(" -> Evolution took ", "{0:.4e}".format((t[-1] - t[0])/ms.year_to_pc), " yrs")
        if accretion:
            return Classic.Evolution(sp, t, R, m2=m2, msg=Int.message)
        return Classic.Evolution(sp, t, R, msg=Int.message)


    def evolve_elliptic_binary(sp, a_0, e_0, a_fin=0., t_0=0., t_fin=None, acc=1e-8, verbose = 1, accretion=False):
        """
        The function evolves the coupled differential equations of the semimajor axis and eccentricity of the Keplerian orbits of the inspiralling system
            by tracking orbital energy and angular momentum loss due  to gravitational wave radiation, dynamical friction and possibly accretion

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a_0  (float)    : The initial semimajor axis
            e_0  (float)    : The initial eccentricity
            a_fin (float)   : The semimajor axis at which to stop evolution
            t_0    (float)  : The initial time
            t_fin  (float)  : The time until the system should be evolved, if None then the estimated coalescence time will be used
            acc    (float)  : An accuracy parameter that is passed to solve_ivp
            verbose (int)   : A parameter describing how verbose the function should be

        Returns:
            ev : Evolution
                An evolution object that contains the results
        """
        def g(e):
            return e**(12./19.)/(1. - e**2) * (1. + 121./304. * e**2)**(870./2299.)

        t_coal =  5./256. * a_0**4/sp.m_total()**2 /sp.m_reduced()
        t_coal = t_coal * 48./19. / g(e_0)**4 * quad(lambda e: g(e)**4 *(1-e**2)**(5./2.) /e/(1. + 121./304. * e**2), 0., e_0, limit=100)[0]   # The inspiral time according to Maggiore (2007)
        if t_fin is None:
            t_fin = 1.1 * t_coal *( 1. - a_fin**4 / a_0**4)

        if a_fin == 0.:
            a_fin = sp.r_isco()
        a_scale = a_fin
        t_scale = t_fin
        if accretion:
            m_scale = sp.m2

        #t_step_max = t_fin/1e4 / t_coal
        t_step_max = np.inf
        if verbose > 0:
            print("Evolving from ", a_0/sp.r_isco(), " to ", a_fin/sp.r_isco(),"r_isco with initial eccentricity ", e_0)

        def dy_dt(t, y, *args):
            sp = args[0]
            t = t*t_scale
            a = y[0]*a_scale
            e = y[1]
            if accretion:
                sp.m2 = y[2] * m_scale

            if verbose > 1:
                tic = time.perf_counter()
            if accretion:
                dm2_dt = Classic.dm2_dt_avg(sp, a, e)
            else:
                dm2_dt = 0.
            da_dt = Classic.da_dt(sp, a, e, accretion=accretion, dm2_dt=dm2_dt)
            de_dt = Classic.de_dt(sp, a, e, da_dt, accretion=accretion, dm2_dt=dm2_dt)

            if verbose > 1:
                toc = time.perf_counter()
                print("t=", t, "a=", a, "da/dt=", da_dt, "e=", e, "de/dt=", de_dt, "m2=", sp.m2, "dm2_dt=", dm2_dt,
                        " elapsed real time: ", toc-tic)


            if accretion:
                return [t_scale/a_scale * da_dt, t_scale* de_dt, t_scale/m_scale * dm2_dt]
            return [t_scale/a_scale * da_dt, t_scale * de_dt]

        fin_reached = lambda t,y, *args: y[0] - a_fin/a_scale           # Give the termination condition such that a = a_fin
        fin_reached.terminal = True

        if accretion:
            y_0 = [a_0/a_scale, e_0, sp.m2/m_scale]
        else:
            y_0 = [a_0/a_scale, e_0]

        Int = solve_ivp(dy_dt, [t_0/t_scale, (t_0+t_fin)/t_scale], y_0, dense_output=True, args=([sp]), events=fin_reached, max_step=t_step_max/t_scale,
                                                                                        method = 'RK45', atol=acc, rtol=acc)

        a = Int.y[0]*a_scale
        e = np.clip(Int.y[1], 1e-50, None)
        #e = Int.y[1]
        t = Int.t*t_scale
        if accretion:
            m2 = Int.y[2]*m_scale

        if verbose > 0:
            print(Int.message)
            print(" -> Evolution took ", "{0:.4e}".format((t[-1] - t[0])/ms.year_to_pc), " yrs")
        if accretion:
            return Classic.Evolution(sp, t, a, e=e, m2=m2, msg=Int.message)
        return Classic.Evolution(sp, t, a, e=e, msg=Int.message)


class HaloModel:
    """
    A class bundling the functions to simulate an inspiral according to the HaloFeedback Model given by https://arxiv.org/abs/2002.12811.pdf
    This class needs an instant, as it calculates a grid for the elliptic function evaluation on the fly

    Attributes:
        N_b (int): The grid size of the impact parameter b
        b_min (float) : The minimum impact parameter b, given by an approximate neutron star size
        sp (SystemProp) : The object describing the information of the inspiral system. sp.halo needs to be of DynamicalSS type
        m_grid (np.ndarray) : The grid of parameter m on which the elliptic function is calculated
        phi_grid (np.ndarray) : The grid of angle phi on which the elliptic function is calculated
        ell_grid (np.ndarrya) : The result of the elliptic function on the grid
    """
    N_b = 50
    b_min = 15e3 * ms.m_to_pc   # 15 km in pc


    def __init__(self, sp):
        """
        The constructor for the HaloModel class

        Parameters:
            sp : SystemProp
                The object describing the inspiral system. sp.halo needs to be of DynamicalSS type
        """
        self.sp = sp
        self.m_grid = np.array([])


    def elliptic_function(m, phi):
        """
        The function calculates the incomplete elliptic integral of the second kind with parameters m and phi with the help of scipy.special.ellipeinc

        Parameters:
            m : array_like
                The parameter m of the elliptic integral
            phi : array_like
                The angle phi of the elliptic integral
        Returns:
            out : float
                The value of the incomplete elliptic integral of the second kind
        """
        N = np.zeros(np.shape(m))
        mask = (m <= 1.)
        invmask = ~mask
        if np.sum(mask) > 0:
            N[mask] = ellipeinc(phi[mask], m[mask])
        if np.sum(invmask) > 0:
            beta = np.arcsin(np.clip(np.sqrt(m[invmask]) * np.sin(phi[invmask]), 0., 1.))
            N[invmask] = np.sqrt(m[invmask]) * ellipeinc(beta, 1./m[invmask]) + ((1. - m[invmask]) / np.sqrt(m[invmask])) * ellipkinc(beta, 1./m[invmask])
        return N


    def elliptic_term_interp(self, m, phi1, phi2):
        """
        The function returns the difference of the elliptic integrals
            E(phi1, m) - E(phi2, m)
        as needed by eq (D5) from https://arxiv.org/abs/2002.12811.pdf
        To obtain the result, ell_grid is interpolated by means of the m_grid, phi_grid that the class saves.
            If the parameter m is outside of the grid (or not initialized), the corresponding ell_grid values are calculated by elliptic_function
            and the grids are appended

        Parameters:
            m : array_like
                The parameters m of the elliptic integral term
            phi1 : array_like
                The angles phi1 of the elliptic integral term
            phi2 : array_like
                The angles phi2 of the elliptic integral term

        Returns:
            out : np.ndarray
                The result of the elliptic integral term
        """
        if len(m) < 1:
            return np.array([])
        phi = np.append(phi1, phi2)
        n_per_decade = 6
        if len(self.m_grid) == 0:   # if the grid is empty it needs to be initialized
            self.m_grid = np.append(0., np.geomspace(1e-5, max(1.001*np.max(m), 10.), n_per_decade * 6 * max(int(np.log10(np.max(m))), 1)))
            self.phi_grid =  np.append(0., np.geomspace( 1e-5, 1.001*np.pi/2., 6*n_per_decade))
            self.mphi_grid = np.array(np.meshgrid(self.m_grid, self.phi_grid)).T.reshape(-1,2)
            self.ell_grid = HaloModel.elliptic_function(self.mphi_grid[:,0], self.mphi_grid[:,1])
        else:
            if np.max(self.m_grid) < np.max(m): # if the grid is insufficient for the m values it needs to be appended
                n_add = max(int(n_per_decade* np.log10(np.max(m) / np.max(self.m_grid))), 2)
                add_to_m_grid = np.geomspace(np.max(self.m_grid) * (1. + 1./float(n_add)) , 1.01*np.max(m), n_add)
                #print("trying to add ", n_add, " values to the m_grid")
                grid = np.array(np.meshgrid(add_to_m_grid, self.phi_grid)).T.reshape(-1,2)
                ell = HaloModel.elliptic_function(grid[:,0], grid[:,1])
                self.m_grid = np.append(self.m_grid, add_to_m_grid)
                self.mphi_grid = np.array(np.meshgrid(self.m_grid, self.phi_grid)).T.reshape(-1,2)
                self.ell_grid = np.append(self.ell_grid, ell)

        ell_interp = griddata(self.mphi_grid, self.ell_grid, np.stack((np.append(m, m), phi), axis=-1),  method='cubic', fill_value=0.) # now we can interpolate the terms
        return ell_interp[:len(phi1)] - ell_interp[len(phi1):]


    def scatter_probability(self, R, Eps, DeltaEps, b_star, v_0, b_90, v_cut=None):
        """
        The function calculates the scattering probability as given by eq (4.12) of https://arxiv.org/abs/2002.12811.pdf

        Parameters:
            R : float
                The radius of the circular orbit of the smaller mass m2
            Eps : np.ndarray
                The meshgrid of relative energy
            DeltaEps : np.ndarray
                The meshgrid of the change of relative energy
            b_star : np.ndarry
                The meshgrid of the impact parameter
            v_0 : float
                The orbital speed of the smaller mass m2
            b_90 : float
                The impact parameter that produces a 90deg deflection
            v_cut : float
                The cut velocity, the maximum velocity DM particles can have to scatter with the smaller object. If None, it is the maximum velocity at the given orbital radius R

        Returns:
            out : np.ndarray
                The scattering probability
        """
        P = np.zeros(np.shape(Eps))

        if v_cut is None:
            v_cut = np.sqrt(2.*self.sp.halo.potential(R))
        else:
            v_cut = np.clip(v_cut, 0., np.sqrt(2.*self.sp.halo.potential(R)))

        g = self.sp.halo.stateDensity(Eps)

        #r_eps = self.sp.m1 / Eps
        r_eps = self.sp.halo.r_of_Eps(Eps)
        #r_cut = self.sp.m1 / (Eps + 0.5* v_cut**2 )
        r_cut = self.sp.halo.r_of_Eps(Eps + 0.5*v_cut**2)

        alpha1 = np.arccos( np.clip(R/b_star * (1. - R/r_eps),  -1., 1.) )
        alpha2 = np.arccos( np.clip(R/b_star * (1. - R/r_cut), -1., 1.))
        m = 2.* b_star/R / (1. - R/r_eps + b_star/R)

        mask = (Eps > ( self.sp.halo.potential(R) * (1. - b_star/R) - 1./2. * v_cut**2 )) & ( Eps < self.sp.halo.potential(R)*(1. + b_star/R))  & (m > 0.) & (alpha2 > alpha1)

        '''  # Uncomment to test the performance of the elliptic term interpolation
        tic = time.perf_counter()
        ellipticIntegral0 = HaloModel.elliptic_function(m[mask], (np.pi-alpha1[mask])/2.) - HaloModel.elliptic_function(m[mask], (np.pi-alpha2[mask])/2.)
        toc = time.perf_counter()
        t0 = toc-tic

        tic = time.perf_counter()
        '''
        ellipticIntegral = self.elliptic_term_interp( m[mask], (np.pi-alpha1[mask])/2., (np.pi-alpha2[mask])/2.)
        '''
        toc = time.perf_counter()
        t = toc-tic
        dif = ellipticIntegral - ellipticIntegral0
        print("e=", ellipticIntegral, "t(e) = ", t, "e0=",ellipticIntegral0, " t(e0)=", t0, "max[e-e0]=", np.max(np.abs(dif)),
                        "max[e/e0-1=]", np.max(np.abs(dif/ellipticIntegral)), "(at m=", m[mask][np.argmax(np.abs(dif/ellipticIntegral))], "phi1=", (np.pi - alpha1[mask])[np.argmax(np.abs(dif/ellipticIntegral))]/2.,
                         ") avg[e/e0-1=]", np.mean(np.abs(dif/ellipticIntegral)))
        '''

        P[mask] = (
                4.*np.pi**2 * R
                / g[mask]
                * (b_90 / v_0)**2
                * 2.* np.sqrt(2.* self.sp.halo.potential(R))
                * (1. + b_star[mask]**2 / b_90**2)**2
                * np.sqrt(1. - R/r_eps[mask] + b_star[mask]/R)
                * ellipticIntegral
                )
        return P


    def P_DeltaEps(v, DeltaEps, b_90, b_min, b_max):
        return 2. * b_90**2 * v**2 / DeltaEps**2 / (b_max**2 - b_min**2)


    def dfHalo_dt(self, R, v_cut=None, t_scale=None):
        """
        The function calculates the secular time derivative of the phase space distribution as given by eq (4.7) of https://arxiv.org/abs/2002.12811.pdf

        If t_scale is given, there is a correction applied to the second part of eq (4.7) such that it cannot be bigger than the first part on the given time scale t_scale.
            This is useful for integration with large timesteps.

        Parameters:
            R : float
                The radius of the circular orbit of the smaller mass m2
            v_cut : float
                The cut velocity, the maximum velocity DM particles can have to scatter with the smaller object. If None, it is the maximum velocity at the given orbital radius R
            t_scale : float
                The timescale on which phase space volume should be conserved

        Returns:
            out : np.ndarray
                The secular time derivative of the phase space distribution
        """
        N_b = HaloModel.N_b
        Eps_grid = self.sp.halo.Eps_grid
        f_grid = self.sp.halo.f_grid
        T_orb = 2.*np.pi / self.sp.omega_s_approx(R)
        v_0 = R* self.sp.omega_s_approx(R)
        b_90 = self.sp.m2/v_0**2
        b_min = HaloModel.b_min
        b_max = np.sqrt(self.sp.m1/self.sp.m2) * np.sqrt(b_90**2 + b_min**2)
        b_grid = np.geomspace(b_min, b_max, N_b)[::-1] # invert so that DeltaEps is ascending

        b_star, Eps = np.meshgrid( b_grid, Eps_grid)
        DeltaEps =  2.*v_0**2 / (1. + (b_star / b_90)**2)

        P_minus  = self.scatter_probability(R, Eps, DeltaEps, b_star, v_0, b_90, v_cut)

        EpspDeltaEps = Eps + DeltaEps
        P_plus = self.scatter_probability(R, EpspDeltaEps, DeltaEps, b_star, v_0, b_90, v_cut)

        norm = simps(HaloModel.P_DeltaEps(v_0, DeltaEps[0], b_90, b_min, b_max), x=DeltaEps[0])

        # Plots the scattering probability on a 2D grid
        '''
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot_wireframe(np.log10(DeltaEps), np.log10(Eps), np.log10(np.clip(P_minus, 1e-30, None)))
        ax.plot_wireframe(np.log10(DeltaEps), np.log10(EpspDeltaEps), np.log10(np.clip(P_plus, 1e-30, None)), color='orange')
        #ax.plot( np.log10(DeltaEps[:,0]), [np.log10(self.sp.m1/R)]*len(DeltaEps[:,0]),   np.log10(np.max(P_minus)), color='red')
        #ax.plot( np.log10(DeltaEps[:,0]), [np.log10(self.sp.m1/R +  np.max(DeltaEps))]*len(DeltaEps[:,0]), np.log10(np.max(P_minus)) , color='green')
        ax.set_xticks(np.log10(np.geomspace(np.min(DeltaEps), np.max(DeltaEps), 6)))
        ax.set_xticklabels(np.geomspace(np.min(DeltaEps), np.max(DeltaEps), 6))
        ax.set_ylim(np.log10([np.max([np.min(Eps), 1e-3]), np.max(Eps)]))
        ax.set_yticks(np.log10(np.geomspace(np.max([np.min(Eps), 1e-3]), np.max(Eps), 6)))
        ax.set_yticklabels(np.geomspace(np.max([np.min(Eps), 1e-3]), np.max(Eps), 6))
        ax.set_zticks(np.log10(np.geomspace(1e-5 * np.max(P_minus), np.max(P_minus), 6)))
        ax.set_zlim(np.log10(1e-5 * np.max(P_minus)), np.log10(np.max(P_minus)))
        ax.set_zticklabels(np.geomspace(1e-5 * np.max(P_minus), np.max(P_minus), 6))
        ax.set_xlabel(r'$\Delta\varepsilon$'); ax.set_ylabel(r'$\varepsilon$')
        '''

        dfHalo = np.zeros(np.shape(f_grid))

        # The first term of eq (4.7)
        dfHalo -= f_grid * simps(P_minus, DeltaEps)/T_orb/norm

        # correction calculation to conserve phase space density on a given t_scale
        correction = np.ones(np.shape(f_grid))
        if not t_scale is None:
            correction = np.clip(f_grid / (-dfHalo * t_scale + 1e-50), 0., 1.)
            dfHalo = np.clip(dfHalo, -f_grid/t_scale, 0.)

        # The second term of eq (4.7)
        dfHalo += simps(    (Eps/EpspDeltaEps )**(5./2.)
                                #* np.interp(EpspDeltaEps, Eps_grid, f_grid*correction)
                                * griddata(Eps_grid, f_grid*correction, EpspDeltaEps, method='cubic', fill_value=0.)
                                * P_plus
                        , x=DeltaEps) / T_orb/norm

        return dfHalo


    def dE_orbit_dR(sp, R):
        """
        The function gives the derivative of the orbital energy wrt the orbital radius r
           of the binary with central mass m1 and the smaller mass m2
           for a circular orbit
        This avoids the calulation of the mass in the halo as it is computationally expensive

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            R  (float)      : The radius of the circular orbit

        Returns:
            out : float
                The derivative of the orbital energy wrt to R of the circular orbit
        """
        return sp.m_reduced()/2.*(sp.m_total()/R**2)


    def dE_gw_dt(sp, R):
        """
        The function gives the energy loss due to radiation of gravitational waves
            for a circular orbit with radius R

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            R  (float)      : The radius of the circular orbit

        Returns:
            out : float
                The energy loss due to radiation of gravitational waves
        """
        omega_s = sp.omega_s_approx(R)
        return 32./5. * sp.m_reduced()**2 * R**4 * omega_s**6


    def dE_df_dt(sp, R):
        """
        The function gives the energy loss due to dynamical friction of the smaller object with the dark matter halo
           on a circular orbit with radius R, as given by eq (2.13) in https://arxiv.org/abs/2002.12811.pdf

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            R  (float)      : The radius of the circular orbit

        Returns:
            out : float
                The energy loss due to dynamical friction
        """
        v_0 = sp.omega_s_approx(R)*R
        ln_Lambda = 1./2.*np.log(sp.m1/sp.m2)
        return 4.*np.pi *sp.m2**2 * sp.halo.density(R, v_max=v_0) * ln_Lambda / v_0


    def dR_dt(sp, R):
        """
        The function gives the secular time derivative of the radius for a circular orbit due to gravitational wave emission and dynamical friction
            of the smaller object on a circular orbit with radius R
        The equation is obtained by the relation
            dR/dt = dE/dt / (dE/dR)
        where dR/dt is the energy loss due to gravitational waves emission and dynamical friction

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            R  (float)      : The radius of the circular orbit

        Returns:
            out : float
                The secular time derivative of the semimajor axis
        """
        return -(HaloModel.dE_gw_dt(sp, R) + HaloModel.dE_df_dt(sp, R))/ HaloModel.dE_orbit_dR(sp, R)


    def evolve_circular_binary(self, R_0, R_fin=0., t_0=0., acc=1e-8, verbose = True):
        """
        The function evolves the system of differential equations as described in https://arxiv.org/pdf/2002.12811.pdf
         for the radius R and for the distribution function f on a grid Eps using the solve_ivp function

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system, sp.halo needs to be of type DynamicSS
            R_0  (float)    : The initial orbital radius
            R_fin (float)   : The orbital radius at which to stop evolution
            t_0    (float)  : The initial time
            acc    (float)  : An accuracy parameter that is passed to solve_ivp
            verbose (bool)  : A parameter describing how verbose the function should be

        Returns:
            t : np.ndarray
                The time steps the integration returns. The first is t_0
            R : np.ndarray
                The corresponding radii at the given time steps
            f : np.ndarry
                The distribution function on the grid at the given time steps
        """
        t_coal =  5./256. * R_0**4/self.sp.m_total()**2 /self.sp.m_reduced()
        t_fin = t_coal *( 1. - R_fin**4 / R_0**4)
        Tfin_orb =  2.*np.pi / self.sp.omega_s_approx(R_fin)
        Tini_orb = 2.*np.pi / self.sp.omega_s_approx(R_0)

        t_scale = 1e3 * Tfin_orb
        R_scale = R_fin

        t_step_max = np.inf
        #t_step_max = 1e4 * Tini_orb

        if verbose:
            print("Evolving from ", R_0/self.sp.r_isco(), " to ", R_fin/self.sp.r_isco(),"r_isco, expected to take at most ", t_coal/t_scale) # with maximum step size ", t_step_max)

        def dy_dt(t, y):
            R = y[0]*R_scale
            v_0 = self.sp.omega_s_approx(R)*R
            self.sp.halo.f_grid = y[1:]

            if verbose:
                tic = time.perf_counter()
            dR_dt = t_scale/R_scale * HaloModel.dR_dt(self.sp, R)
            df_dt = t_scale*self.dfHalo_dt(R, v_cut=v_0)
            if verbose:
                toc = time.perf_counter()
                print('t=', t, 'R=', R/self.sp.r_isco(), 'y =', y, np.where(self.sp.halo.f_grid < 0.), 'dR/dt = ', dR_dt*R_scale/self.sp.r_isco(), 'dy_dt = ', df_dt, f" elapsed real time {toc-tic:0.4f} seconds")
            return np.append(dR_dt, df_dt)

        fin_reached = lambda t,y, *args: y[0] - R_fin/R_scale
        fin_reached.terminal = True

        y_0 = np.append(R_0/R_scale, self.sp.halo.f_grid)
        y_0_atol = np.append(acc, self.sp.halo.f_grid * np.sqrt(acc))
        y_0_rtol = acc

        Int = solve_ivp(dy_dt, [t_0/t_scale, (t_0+t_fin)/t_scale], y_0, dense_output=True, events=fin_reached, max_step=t_step_max,
                                                                                        method = 'RK23', atol=y_0_atol, rtol=y_0_rtol)

        R = Int.y[0]*R_scale
        f = np.transpose(Int.y[1:])
        t = Int.t*t_scale

        if verbose:
            print(Int.message)
            print(f" -> Inspiral took {(t[-1] - t[0])/ms.year_to_pc :0.4f} yrs and {len(t) :,} steps")

        return t, R, f


    def evolve_circular_binary_HFK(self, R_0, R_fin=0., t_0=0.):
        """
        The function evolves the system of differential equations as described in https://arxiv.org/pdf/2002.12811.pdf
         for the radius R and for the distribution function f on a grid Eps using a simplified improved Euler integration

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            R_0  (float)    : The initial orbital radius
            R_fin (float)   : The orbital radius at which to stop evolution
            t_0    (float)  : The initial time

        Returns:
            t : np.ndarray
                The time steps the integration returns. The first is t_0
            R : np.ndarray
                The corresponding radii at the given time steps
            f : np.ndarry
                The distribution function on the grid at the given time steps
        """
        t_coal =  5./256. * R_0**4/self.sp.m_total()**2 /self.sp.m_reduced()
        t_fin = t_coal *( 1. - R_fin**4 / R_0**4)
        Tfin_orb =  2.*np.pi / self.sp.omega_s_approx(R_fin)
        Tini_orb = 2.*np.pi / self.sp.omega_s_approx(R_0)

        dt_Torb = 5e3
        N_step = int(t_fin/Tfin_orb/dt_Torb)
        dt = dt_Torb *Tini_orb

        print("Evolving from ", R_0/self.sp.r_isco(), " to ", R_fin/self.sp.r_isco(),"r_isco", " - dt=", dt, " with max steps ", N_step)

        t_list = np.array([t_0]);   t = t_0
        f_list = np.array([self.sp.halo.f_grid]);
        R_list = np.array([R_0]);   R = R_0

        i = 0
        while i < N_step and R > R_fin:
            v_0 = self.sp.omega_s_approx(R)*R

            df1 = dt * self.dfHalo_dt(R, v_cut=v_0, t_scale=dt)
            dr1 = dt * HaloModel.dR_dt(self.sp, R)
            self.sp.halo.f_grid += df1; R += dr1;
            df2 = dt * self.dfHalo_dt(R, v_cut=v_0, t_scale=dt)
            dr2 = dt * HaloModel.dR_dt(self.sp, R)
            self.sp.halo.f_grid += 0.5 * (df2-df1);  R += 0.5*(dr2-dr1);
            t += dt

            print(i, "t=",t, ",dt=", dt, ",R=", R/self.sp.r_isco(), ",f=" , self.sp.halo.f_grid)
            t_list = np.append(t_list, t+dt)
            f_list = np.concatenate((f_list, [self.sp.halo.f_grid]))
            R_list = np.append(R_list, R)
            i+= 1
            T_orb = 2.*np.pi/self.sp.omega_s_approx(R)
            dt = np.min([dt, dt_Torb*T_orb])

        return t_list, R_list, f_list

import numpy as np
from scipy.integrate import solve_ivp, quad

import time
from imripy import constants as c, kepler, merger_system as ms
from . import forces

class Classic:
    """
    A class bundling the functions to simulate an inspiral with basic energy conservation arguments
    This class does not need to be instantiated
    """

    class EvolutionOptions:
        """
        This class allows to modify the behavior of the evolution of the differential equations

        Attributes:
            accuracy : float
                An accuracy parameter that is passed to solve_ivp
            verbose : int
                A verbosity parameter ranging from 0 to 2
            elliptic : bool
                Whether to model the inspiral on eccentric orbits, is set automatically depending on e0 passed to Evolve
            m2_chance : bool
                Whether to evolve the secondary's mass m2 with time (e.g. through accretion)
            dissipativeForces : list of forces.DissipativeForces
                List of the dissipative forces employed during the inspiral
            gwEmissionLoss : bool
                These parameters are for backwards compatibility - Only applies if dissipativeForces=None - then forces.GWLoss is added to the list
            dynamicalFrictionLoss : bool
                These parameters are for backwards compatibility - Only applies if dissipativeForces=None - then forces.GWLoss is added to the list
            considerRelativeVelocities: bool
                Wether to consider the relative velocities between the secondary and the environmental distributions. So far only relevant for accretion disk models
            progradeRotation : bool
                Wether to orbit prograde or retrograde wrt to some other orbiting object - so far only relevant for accretion disk models
            periapsePrecession : bool
                Wether to include precession of the periapse due to relativistic precession and mass precession
            **kwargs : additional parameter
                Will be saved in opt.additionalParameters and will be available throughout the integration

        """
        def __init__(self, accuracy=1e-10, verbose=1, elliptic=True, m2_change=False,
                                    dissipativeForces=None, gwEmissionLoss = True, dynamicalFrictionLoss = True,
                                    considerRelativeVelocities=False, progradeRotation = True,
                                    periapsePrecession = False,
                                    **kwargs):
            self.accuracy = accuracy
            self.verbose = verbose
            self.elliptic = elliptic
            self.m2_change = m2_change
            if dissipativeForces is None:
                dissipativeForces = []
                if gwEmissionLoss:
                    dissipativeForces.append(forces.GWLoss())
                if dynamicalFrictionLoss:
                    dissipativeForces.append(forces.DynamicalFriction())
            self.dissipativeForces = dissipativeForces
            self.considerRelativeVelocities = considerRelativeVelocities
            self.progradeRotation = progradeRotation
            self.periapsePrecession = periapsePrecession
            self.additionalParameters = kwargs


        def __str__(self):
            s = "Options: dissipative forces employed {"
            for df in self.dissipativeForces:
                s += str(df) + ", "
            s += "}" + f", accuracy = {self.accuracy:.1e}"
            for key, value in self.additionalParameters.items():
                s += f", {key}={value}"
            return s


    def E_orbit(hs, ko, opt=EvolutionOptions()):
        """
        The function gives the orbital energy of the binary with central mass m1 with the surrounding halo and the smaller mass m2
           for a Keplerian orbit with semimajor axis a and eccentricity e

        # TODO : fix inclusion of mass in halo
        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy of the Keplerian orbit
        """
        return  - ko.m_tot * ko.m_red / ko.a / 2.


    def dE_orbit_da(hs, ko, opt=EvolutionOptions()):
        """
        The function gives the derivative of the orbital energy wrt the semimajor axis a
           of the binary with central mass m1 with the surrounding halo and the smaller mass m2
           for a Keplerian orbit with semimajor axis a and eccentricity e
        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The derivative of the orbital energy wrt to a of the Keplerian orbit
        """
        return ko.m2 * hs.mass(ko.a) / 2. / ko.a**2  * ( 1. - ko.a*hs.dmass_dr(ko.a) / hs.mass(ko.a) )

    def L_orbit(hs, ko, opt=EvolutionOptions()):
        """
        The function gives the angular momentum of the binary with central mass m1 with the surrounding halo and the smaller mass m2
           for a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum of the Keplerian orbit
        """
        L = np.sqrt(ko.a * (1-ko.e**2) * ko.m_tot * ko.m_red**2 )
        return L if ko.prograde else -L
        #return np.sqrt( -(1. - e**2) * sp.m_reduced(a)**3 * sp.m_total(a)**2 / 2. / Classic.E_orbit(sp, a, e))


    def dE_dt(hs, ko, opt=EvolutionOptions()):
        """
        The function gives the total energy loss of the orbiting small black hole due to the dissipative effects
           on a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The total energy loss
        """
        dE_dt_tot = 0.
        dE_dt_out = ""
        for df in opt.dissipativeForces:
            dE_dt = df.dE_dt(hs, ko, opt)
            dE_dt_tot += dE_dt
            if opt.verbose > 2:
                dE_dt_out += f"dE({df.name})/dt:{dE_dt}, "

        if opt.verbose > 2:
            print(dE_dt_out)
        return  dE_dt_tot


    def dL_dt(hs, ko, opt=EvolutionOptions()):
        """
        The function gives the total angular momentum loss of the secondary object
            on a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The total angular momentum loss
        """
        dL_dt_tot = 0.
        dL_dt_out = ""
        for df in opt.dissipativeForces:
            dL_dt = df.dL_dt(hs, ko, opt)
            dL_dt_tot += dL_dt
            if opt.verbose > 2:
                dL_dt_out += f"dL({df.name})/dt:{dL_dt}, "

        if opt.verbose > 2:
            print(dL_dt_out)
        return  dL_dt_tot


    def dm2_dt(hs, ko, opt=EvolutionOptions()):
        """
        The function gives the secular time derivative of the mass of the secondary m2 due to accretion of a halo
            of the smaller object on a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            dm2_dt : float
                The secular time derivative of the mass of the secondary

        """
        dm2_dt_tot = 0.
        dm2_dt_out = ""
        for df in opt.dissipativeForces:
            if not df.m2_change:
                continue
            dm2_dt = df.dm2_dt_avg(hs, ko, opt)
            dm2_dt_tot += dm2_dt
            if opt.verbose > 2:
                dm2_dt_out += f"{df.name}:{dm2_dt}, "

        if opt.verbose > 2:
            print(dm2_dt_out)

        return dm2_dt_tot


    def da_dt(hs, ko, opt=EvolutionOptions(), return_dE_dt=False):
        """
        The function gives the secular time derivative of the semimajor axis a (or radius for a circular orbit) due to gravitational wave emission and dynamical friction
            of the smaller object on a Keplerian orbit with semimajor axis a and eccentricity e
        The equation is obtained by the relation
            E = -m_1 * m_2 / 2a

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations
            dE_dt (bool)    : Whether to return dE_dt in addition to da_dt, to save computation time

        Returns:
            da_dt : float
                The secular time derivative of the semimajor axis
            dE_dt : float
                The secular time derivative of the orbital energy
        """
        dE_dt = Classic.dE_dt(hs, ko, opt)
        dE_orbit_da = Classic.dE_orbit_da(hs, ko, opt)

        if return_dE_dt:
            return dE_dt / dE_orbit_da, dE_dt

        return    ( dE_dt / dE_orbit_da )


    def de_dt(hs, ko, dE_dt=None, opt=EvolutionOptions()):
        """
        The function gives the secular time derivative of the eccentricity due to gravitational wave emission and dynamical friction
            of the smaller object on a Keplerian orbit with semimajor axis a and eccentricity e
        The equation is obtained by the time derivative of the relation
            e^2 = 1 + 2EL^2 / m_total^2 / m_reduced^3
           as given in Maggiore (2007)

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            dE_dt (float)   : Optionally, the dE_dt value if it was computed previously
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The secular time derivative of the eccentricity
        """
        if ko.e <= 0. or not opt.elliptic:
            return 0.

        dE_dt = Classic.dE_dt(hs, ko, opt) if dE_dt is None else dE_dt
        E = Classic.E_orbit(hs, ko, opt)
        dL_dt = Classic.dL_dt(hs, ko, opt)
        L = Classic.L_orbit(hs, ko, opt)

        if opt.verbose > 2:
            print("dE_dt/E=", dE_dt/E, "2dL_dt/L=", 2.*dL_dt/L, "diff=", dE_dt/E + 2.*dL_dt/L )

        return - (1.-ko.e**2)/2./ko.e *(  dE_dt/E + 2. * dL_dt/L   )


    def dperiapse_angle_dt(hs, ko, opt=EvolutionOptions()):
        """
        The function gives the secular time derivative of the argument of periapse due to relativistic (Schwarzschild) precession
            and the mass precession of the halo mass
        The relativistic precession is given by the first term of equation (11) and the mass precession part is given by a generalization of
            equation (10) of https://arxiv.org/pdf/2111.13514.pdf

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The secular time derivative of the periapse angle
        """
        a = ko.a; e = ko.e
        T = 2.*np.pi * np.sqrt(a**3/ko.m_tot)
        # relativistic precession
        dperiapse_angle_dt_rp = 6.*np.pi * ko.m_tot / a / (1-e**2) / T

        # mass precession
        def integrand(phi):
            r = a*(1. - e**2)/(1. + e*np.cos(phi))
            return np.cos(phi) * hs.halo.mass(r) / ko.m_tot
        dperiapse_angle_dt_m =  1./e / T  * quad(integrand, 0., 2.*np.pi, limit = 100)[0]

        return dperiapse_angle_dt_rp + dperiapse_angle_dt_m


    def dinclination_angle_dt(hs, ko, opt=EvolutionOptions()):
        """


        """
        di_dt_tot = 0.
        di_dt_out = ""
        for df in opt.dissipativeForces:
            di_dt = df.dinclination_angle_dt(hs, ko, opt)
            di_dt_tot += di_dt
            if opt.verbose > 2:
                di_dt_out += f"di({df.name})/dt:{di_dt}, "

        if opt.verbose > 2:
            print(di_dt_out)
        return di_dt_tot



    def handle_args(hs, ko, a_fin, t_0, t_fin, opt):
        opt.elliptic = ko.e > 0.

        # calculate relevant timescales
        def g(e):
            return e**(12./19.)/(1. - e**2) * (1. + 121./304. * e**2)**(870./2299.)

        t_coal =  5./256. * ko.a**4 / ko.m_tot**2 / ko.m_red
        if opt.elliptic:
            t_coal = t_coal * 48./19. / g(ko.e)**4 * quad(lambda e: g(e)**4 *(1-e**2)**(5./2.) /e/(1. + 121./304. * e**2), 0., ko.e, limit=100)[0]   # The inspiral time according to Maggiore (2007)

        if t_fin is None:
            t_fin = 1.2 * t_coal *( 1. - a_fin**4 / ko.a**4)    # This is the time it takes with just gravitational wave emission

        if a_fin == 0.:
            a_fin = hs.r_isco     # Stop evolution at r_isco

        ko.prograde = opt.progradeRotation # Check?
        return hs, ko, a_fin, t_0, t_fin, opt


    def Evolve(hs, ko, a_fin=0., t_0=0., t_fin=None, opt=EvolutionOptions()):
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
            opt   (EvolutionOptions) : Collecting the options for the evolution of the differential equations

        Returns:
            ev : Evolution
                An evolution object that contains the results
        """
        hs, ko, a_fin, t_0, t_fin, opt = Classic.handle_args(hs, ko, a_fin, t_0, t_fin, opt)

        # set scales to get rescale integration variables
        a_scale = ko.a
        t_scale = t_fin
        m_scale = ko.m2 if opt.m2_change else 1.

        t_step_max = t_fin
        if opt.verbose > 0:
            print("Evolving from ", ko.a/hs.r_isco, " to ", a_fin/hs.r_isco,"r_isco ", ("with initial eccentricity " + str(ko.e)) if opt.elliptic else " on circular orbits", " with ", opt)

        # Define the evolution function
        def dy_dt(t, y, *args):
            hs = args[0]; opt = args[1]
            t = t*t_scale

            # Unpack array
            ko.a, ko.e, ko.m2, ko.periapse_angle = y
            ko.a *= a_scale; ko.m2 = ko.m2 * m_scale if opt.m2_change else ko.m2

            if opt.verbose > 1:
                tic = time.perf_counter()

            da_dt, dE_dt = Classic.da_dt(hs, ko, opt=opt, return_dE_dt=True)
            de_dt = Classic.de_dt(hs, ko, dE_dt=dE_dt, opt=opt) if opt.elliptic else 0.
            dm2_dt = Classic.dm2_dt(hs, ko, opt) if opt.m2_change else 0.
            dperiapse_angle_dt = Classic.dperiapse_angle_dt(hs, ko, opt) if opt.periapsePrecession else 0.

            if opt.verbose > 1:
                toc = time.perf_counter()
                print(rf"Step: t={t : 0.1e}, a={ko.a : 0.1e}({ko.a/hs.r_isco : 0.1e} r_isco)({ko.a/a_fin:0.1e} a_fin), da/dt={da_dt : 0.1e} \\n"
                          +  rf"\\t e={ko.e : 0.1e}, de/dt={ de_dt : 0.1e}, m2={ko.m2:0.1e}, dm2/dt={dm2_dt:0.1e}\\n"
                           + rf"\\t elapsed real time: { toc-tic } s")


            dy = np.array([da_dt/a_scale, de_dt, dm2_dt/m_scale, dperiapse_angle_dt])
            return dy * t_scale

        # Termination condition
        fin_reached = lambda t,y, *args: y[0] - a_fin/a_scale
        fin_reached.terminal = True

        inside_BH = lambda t,y, *args: y[0]*a_scale * (1. - y[1]) - 8*hs.m1  # for a(1-e) < 8m_1
        inside_BH.terminal = True

        # Initial conditions
        y_0 = np.array([ko.a / a_scale, ko.e, ko.m2/m_scale, ko.periapse_angle])

        # Evolve
        tic = time.perf_counter()
        Int = solve_ivp(dy_dt, [t_0/t_scale, (t_0+t_fin)/t_scale], y_0, dense_output=True, args=(hs,opt), events=[fin_reached, inside_BH], max_step=t_step_max/t_scale,
                                                                                        method = 'RK45', atol=opt.accuracy, rtol=opt.accuracy)
        toc = time.perf_counter()

        # Collect results
        ev = Classic.EvolutionResults( hs, opt,
                                            Int.t*t_scale,
                                            Int.y[2]*m_scale if opt.m2_change else ko.m2,
                                            Int.y[0]*a_scale,
                                            Int.y[1] if opt.elliptic else np.zeros(np.shape(Int.y[0])),
                                            Int.y[3] if opt.periapsePrecession else ko.periapse_angle,
                                            inclination_angle=ko.inclination_angle,
                                            longitude_an=ko.longitude_an,
                                            msg=Int.message)

        if opt.verbose > 0:
            print(Int.message)
            print(f" -> Evolution took {toc-tic:.4f}s")

        return ev

    def Evolve_old(sp, a_0, e_0=0., a_fin=0., t_0=0., t_fin=None, opt=EvolutionOptions()):
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
            opt   (EvolutionOptions) : Collecting the options for the evolution of the differential equations

        Returns:
            ev : Evolution
                An evolution object that contains the results
        """
        hs = ms.HostSystem(sp.m1, halo=sp.halo, D_l = sp.D, includeHaloInTotalMass=sp.includeHaloInTotalMass)
        ko = kepler.KeplerOrbit(hs, sp.m2, a_0, e=e_0, periapse_angle=sp.pericenter_angle, inclination_angle=sp.inclination_angle, prograde=opt.progradeRotation)
        return Classic.Evolve(hs, ko, a_fin=a_fin, t_0=t_0, t_fin=t_fin, opt=opt)


    class EvolutionResults(kepler.KeplerOrbit):
        """
        This class keeps track of the evolution of an inspiral.
        This is basically a KeplerOrbit object but with arrays as values for the parameters
        And additionally, the time, the inspiral options and msg

        Attributes:
            hs : merger_system.HostSystem
                The host system
            opt : Classic.EvolutionOptions
                The options used during the evolution
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
        """
        def __init__(self, hs, options, t, m2, a, e=0., periapse_angle=0., inclination_angle=0., longitude_an=0., msg=None):
            self.hs = hs
            self.options = options
            self.msg=msg
            self.t = t
            self.m2 = m2
            self.a = a
            self.e = e
            self.periapse_angle = periapse_angle
            self.inclination_angle = inclination_angle
            self.longitude_an = longitude_an
            if not options.elliptic:
                self.R = a

    def from_xy_plane_to_rhophi_plane(self, v):
        raise NotImplementedError

    def from_rhophi_plane_to_xy_plane(self, v):
        raise NotImplementedError

    def from_orbital_xy_plane_to_fundamental_xy_plane(self, x):
        raise NotImplementedError

    def from_fundamental_xy_plane_to_orbital_xy_plane(self, x):
        raise NotImplementedError

    def get_orbital_vectors_in_orbital_xy_plane(self, phi):
        raise NotImplementedError

    def get_orbital_vectors_in_fundamental_xy_plane(self, phi):
        raise NotImplementedError

    def get_orbital_vectors(self, phi):
        raise NotImplementedError

    def get_orbital_parameters(self, phi):
        raise NotImplementedError

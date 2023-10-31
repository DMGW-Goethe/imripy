import numpy as np
from scipy.integrate import solve_ivp, quad

import time
import imripy.constants as c
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


    def E_orbit(sp, a, e=0., opt=EvolutionOptions()):
        """
        The function gives the orbital energy of the binary with central mass m1 with the surrounding halo and the smaller mass m2
           for a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit, default is 0 - a circular orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy of the Keplerian orbit
        """
        return  - sp.m_total(a)*sp.m_reduced(a) / a / 2.


    def dE_orbit_da(sp, a, e=0., opt=EvolutionOptions()):
        """
        The function gives the derivative of the orbital energy wrt the semimajor axis a
           of the binary with central mass m1 with the surrounding halo and the smaller mass m2
           for a Keplerian orbit with semimajor axis a and eccentricity e
        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The derivative of the orbital energy wrt to a of the Keplerian orbit
        """
        return sp.m2 * sp.mass(a) / 2. / a**2  * ( 1.  - a*sp.dmass_dr(a)/sp.mass(a) )

    def L_orbit(sp, a, e, opt=EvolutionOptions()):
        """
        The function gives the angular momentum of the binary with central mass m1 with the surrounding halo and the smaller mass m2
           for a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum of the Keplerian orbit
        """
        L = np.sqrt(a * (1-e**2) * sp.m_total(a) * sp.m_reduced(a)**2 )
        return L if opt.progradeRotation else -L
        #return np.sqrt( -(1. - e**2) * sp.m_reduced(a)**3 * sp.m_total(a)**2 / 2. / Classic.E_orbit(sp, a, e))


    def dE_dt(sp, a, e=0., opt=EvolutionOptions()):
        """
        The function gives the total energy loss of the orbiting small black hole due to the dissipative effects
           on a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The total energy loss
        """
        dE_dt_tot = 0.
        dE_dt_out = ""
        for df in opt.dissipativeForces:
            dE_dt = df.dE_dt(sp, a, e, opt)
            dE_dt_tot += dE_dt
            if opt.verbose > 2:
                dE_dt_out += f"dE({df.name})/dt:{dE_dt}, "

        if opt.verbose > 2:
            print(dE_dt_out)
        return  dE_dt_tot


    def dL_dt(sp, a, e, opt=EvolutionOptions()):
        """
        The function gives the total angular momentum loss of the secondary object
            on a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The total angular momentum loss
        """
        dL_dt_tot = 0.
        dL_dt_out = ""
        for df in opt.dissipativeForces:
            dL_dt = df.dL_dt(sp, a, e, opt)
            dL_dt_tot += dL_dt
            if opt.verbose > 2:
                dL_dt_out += f"dL({df.name})/dt:{dL_dt}, "

        if opt.verbose > 2:
            print(dL_dt_out)
        return  dL_dt_tot


    def dm2_dt(sp, a, e=0., opt=EvolutionOptions()):
        """
        The function gives the secular time derivative of the mass of the secondary m2 due to accretion of a halo
            of the smaller object on a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
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
            dm2_dt = df.dm2_dt_avg(sp, a, e, opt)
            dm2_dt_tot += dm2_dt
            if opt.verbose > 2:
                dm2_dt_out += f"{df.name}:{dm2_dt}, "

        if opt.verbose > 2:
            print(dm2_dt_out)

        return dm2_dt_tot


    def da_dt(sp, a, e=0., opt=EvolutionOptions(), return_dE_dt=False):
        """
        The function gives the secular time derivative of the semimajor axis a (or radius for a circular orbit) due to gravitational wave emission and dynamical friction
            of the smaller object on a Keplerian orbit with semimajor axis a and eccentricity e
        The equation is obtained by the relation
            E = -m_1 * m_2 / 2a

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations
            dE_dt (bool)    : Whether to return dE_dt in addition to da_dt, to save computation time

        Returns:
            da_dt : float
                The secular time derivative of the semimajor axis
            dE_dt : float
                The secular time derivative of the orbital energy
        """
        dE_dt = Classic.dE_dt(sp, a, e, opt)
        dE_orbit_da = Classic.dE_orbit_da(sp, a, e, opt)

        if return_dE_dt:
            return dE_dt / dE_orbit_da, dE_dt

        return    ( dE_dt / dE_orbit_da )


    def de_dt(sp, a, e, dE_dt=None, opt=EvolutionOptions()):
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
            dE_dt (float)   : Optionally, the dE_dt value if it was computed previously
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The secular time derivative of the eccentricity
        """
        if e <= 0. or not opt.elliptic:
            return 0.

        dE_dt = Classic.dE_dt(sp, a, e, opt) if dE_dt is None else dE_dt
        E = Classic.E_orbit(sp, a, e, opt)
        dL_dt = Classic.dL_dt(sp, a, e, opt)
        L = Classic.L_orbit(sp, a, e, opt)

        if opt.verbose > 2:
            print("dE_dt/E=", dE_dt/E, "2dL_dt/L=", 2.*dL_dt/L, "diff=", dE_dt/E + 2.*dL_dt/L )

        return - (1.-e**2)/2./e *(  dE_dt/E + 2. * dL_dt/L   )


    def dperiapse_angle_dt(sp, a, e, opt=EvolutionOptions()):
        """
        The function gives the secular time derivative of the argument of periapse due to relativistic (Schwarzschild) precession
            and the mass precession of the halo mass
        The relativistic precession is given by the first term of equation (11) and the mass precession part is given by a generalization of
            equation (10) of https://arxiv.org/pdf/2111.13514.pdf

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The secular time derivative of the pericenter angle
        """
        T = 2.*np.pi * np.sqrt(a**3/sp.m_total())
        # relativistic precession
        dperiapse_angle_dt_rp = 6.*np.pi * sp.m_total() / a / (1-e**2) / T

        # mass precession
        def integrand(phi):
            r = a*(1. - e**2)/(1. + e*np.cos(phi))
            return np.cos(phi) * sp.halo.mass(r) / sp.m_total()
        dperiapse_angle_dt_m =  1./e / T  * quad(integrand, 0., 2.*np.pi, limit = 100)[0]

        return dperiapse_angle_dt_rp + dperiapse_angle_dt_m

    class EvolutionResults:
        """
        This class keeps track of the evolution of an inspiral.

        Attributes:
            sp : merger_system.SystemProp
                The system properties used in the evolution
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
        def __init__(self, sp, options, t, a, msg=None):
            self.sp = sp
            self.options = options
            self.msg=msg
            self.t = t
            self.a = a
            if not options.elliptic:
                self.e = np.zeros(np.shape(t))
                self.R = a



    def Evolve(sp, a_0, e_0=0., a_fin=0., t_0=0., t_fin=None, opt=EvolutionOptions()):
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
        opt.elliptic = e_0 > 0.

        # calculate relevant timescales
        def g(e):
            return e**(12./19.)/(1. - e**2) * (1. + 121./304. * e**2)**(870./2299.)

        t_coal =  5./256. * a_0**4/sp.m_total()**2 /sp.m_reduced()
        if opt.elliptic:
            t_coal = t_coal * 48./19. / g(e_0)**4 * quad(lambda e: g(e)**4 *(1-e**2)**(5./2.) /e/(1. + 121./304. * e**2), 0., e_0, limit=100)[0]   # The inspiral time according to Maggiore (2007)

        if t_fin is None:
            t_fin = 1.2 * t_coal *( 1. - a_fin**4 / a_0**4)    # This is the time it takes with just gravitational wave emission

        if a_fin == 0.:
            a_fin = sp.r_isco()     # Stop evolution at r_isco

        # set scales to get rescale integration variables
        a_scale = a_0
        t_scale = t_fin
        m_scale = sp.m2 if opt.m2_change else 1.

        t_step_max = t_fin
        if opt.verbose > 0:
            print("Evolving from ", a_0/sp.r_isco(), " to ", a_fin/sp.r_isco(),"r_isco ", ("with initial eccentricity " + str(e_0)) if opt.elliptic else " on circular orbits", " with ", opt)

        # Define the evolution function
        def dy_dt(t, y, *args):
            sp = args[0]; opt = args[1]
            t = t*t_scale

            # Unpack array
            a, e, m2, periapse_angle = y
            a *= a_scale; sp.m2 = m2 * m_scale if opt.m2_change else sp.m2

            if opt.verbose > 1:
                tic = time.perf_counter()

            da_dt, dE_dt = Classic.da_dt(sp, a, e, opt=opt, return_dE_dt=True)
            de_dt = Classic.de_dt(sp, a, e, dE_dt=dE_dt, opt=opt) if opt.elliptic else 0.
            dm2_dt = Classic.dm2_dt(sp, a, e, opt) if opt.m2_change else 0.
            dperiapse_angle_dt = Classic.dperiapse_angle_dt(sp, a, e, opt) if opt.periapsePrecession else 0.

            if opt.verbose > 1:
                toc = time.perf_counter()
                print(rf"Step: t={t : 0.1e}, a={a : 0.1e}({a/sp.r_isco() : 0.1e} r_isco)({a/a_fin:0.1e} a_fin), da/dt={da_dt : 0.1e} \\n"
                          +  rf"\\t e={e : 0.1e}, de/dt={ de_dt : 0.1e}, m2={sp.m2:0.1e}, dm2/dt={dm2_dt:0.1e}\\n"
                           + rf"\\t elapsed real time: { toc-tic } s")


            dy = np.array([da_dt/a_scale, de_dt, dm2_dt/m_scale, dperiapse_angle_dt])
            return dy * t_scale

        # Termination condition
        fin_reached = lambda t,y, *args: y[0] - a_fin/a_scale
        fin_reached.terminal = True

        inside_BH = lambda t,y, *args: y[0]*a_scale * (1. - y[1]) - 8*sp.m1  # for a(1-e) < 8m_1
        inside_BH.terminal = True

        # Initial conditions
        y_0 = np.array([a_0 / a_scale, e_0, sp.m2/m_scale, sp.pericenter_angle])

        # Evolve
        tic = time.perf_counter()
        Int = solve_ivp(dy_dt, [t_0/t_scale, (t_0+t_fin)/t_scale], y_0, dense_output=True, args=(sp,opt), events=[fin_reached, inside_BH], max_step=t_step_max/t_scale,
                                                                                        method = 'RK45', atol=opt.accuracy, rtol=opt.accuracy)
        toc = time.perf_counter()

        # Collect results
        t = Int.t*t_scale
        a = Int.y[0]*a_scale;
        ev = Classic.EvolutionResults(sp, opt, t, a, msg=Int.message)
        ev.e = Int.y[1] if opt.elliptic else np.zeros(np.shape(ev.t))
        ev.m2 = Int.y[2]*m_scale if opt.m2_change else sp.m2;
        ev.periapse_angle = Int.y[3] if opt.periapsePrecession else sp.pericenter_angle

        if opt.verbose > 0:
            print(Int.message)
            print(f" -> Evolution took {toc-tic:.4f}s")

        return ev




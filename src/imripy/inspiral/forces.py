import numpy as np
from scipy.integrate import quad, quad_vec, odeint, simpson
from scipy.interpolate import interp1d, LinearNDInterpolator, CloughTocher2DInterpolator
import collections

import imripy.constants as c
import imripy.halo
from imripy import kepler

import matplotlib.pyplot as plt # for debugging only



class DissipativeForce:
    """
    This is a model class from which the dissipative forces should be derived
    It contains helper function that calculate the relative velocities during an orbit.
    Also, it defines standard integration function that calculate dE/dt, dL/dt from an arbitrary F, which is to be defined by the class
    Of course, these functions can be overriden in case of an analytic solution (e.g. gravitational wave losses)

    Also contains functions for possible accretion of the secondary dm2_dt.
    Should be easily extendible.
    """
    name = "DissipativeForce"
    m2_change = False


    def F(self, hs, ko, r, v, opt):
        """
        Placeholder function that models the dissipative force strength.

        Parameters:
            hs (HostSystem) : The object describing the properties of the host system
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            r  (np.ndarray) : The position of the secondary in the XYZ fundamental plane
            v  (np.ndarray) : The velocity vector of the secondary in the XYZ fundamental plane
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : np.ndarray
                The vector of the dissipative force in the XYZ fundamental plane
        """
        pass


    def dE_dt(self, hs, ko, opt):
        """
        The function calculates the energy loss due to a force F(r,v) by averaging over a Keplerian orbit
           with semimajor axis a and eccentricity e
        According to https://arxiv.org/abs/1908.10241

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy loss due to accretion
        """
        a = ko.a; e = ko.e
        if  isinstance(ko, (collections.Sequence, np.ndarray)):
            return np.array([self.dE_dt(hs, ko_i, opt) for ko_i in ko])
        if e == 0.:
            #v = hs.omega_s(a)*a
            r, v = ko.get_orbital_vectors(0.)
            F = self.F(hs, ko, r, v, opt)
            F_proj = np.sum(F*v)
            return - F_proj
        else:
            def integrand(phi):
                r, v = ko.get_orbital_vectors(phi)
                F = self.F(hs, ko, r, v, opt)
                F_proj = np.sum(F*v)
                return F_proj / (1.+e*np.cos(phi))**2
            return -(1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    def dL_dt(self, hs, ko, opt):
        """
        The function calculates the angular momentum loss due to a force F(r,v) by averaging over a Keplerian orbit
           with semimajor axis a and eccentricity e
        According to https://arxiv.org/abs/1908.10241

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum loss due to accretion
        """
        a = ko.a; e = ko.e
        def integrand(phi):
            r, v = ko.get_orbital_vectors(phi)
            F = self.F(hs, ko, r, v, opt)
            e_phi = ko.get_orbital_vectors_in_fundamental_xy_plane(phi)[1]
            F_proj = np.sum(F*e_phi) * np.sqrt(np.sum(r*r))
            return F_proj / (1.+e*np.cos(phi))**2
        return -(1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]

    def dinclination_angle_dt(self, hs, ko, opt):
        """
        The function calculates the inclination agnle change due to a force F(r,v)
            calculated with osculating orbital elements

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum loss due to accretion
        """
        a = ko.a; e = ko.e
        Omega = hs.omega_s(a)
        def integrand(phi):
            r, v = ko.get_orbital_vectors(phi)
            F = self.F(hs, ko, r, v, opt)
            e_z = ko.get_orbital_vectors_in_fundamental_xy_plane(phi)[2]
            W = np.sum(F*e_z) / ko.m2
            di_dt = np.sqrt(np.sum(r*r)) * np.cos(ko.periapse_angle + phi) * W / Omega / a**2 / np.sqrt(1-e**2)
            return  di_dt / (1.+e*np.cos(phi))**2
        return -(1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    def dm2_dt(self, hs, ko, r, v, opt):
        """
        Placeholder function that models the mass gain/loss

        Parameters:
            hs (SystemProp) : The object describing the properties of the host system
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            r  (np.ndarray) : The position of the secondary in the XYZ fundamental plane
            v  (np.ndarray) : The velocity vector of the secondary in the XYZ fundamental plane
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The derivative of the secondary mass
        """
        return 0.


    def dm2_dt_avg(self, hs, ko, opt):
        """
        The function gives the mass gain due to accretion of the small black hole inside of a halo
           on a Keplerian orbit with semimajor axis a and eccentricity e
        For a circular orbit the dm2_dt function with the corresponding orbital velocity is used
            for an elliptic orbit the average of the expression is used

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The mass gain of the secondary
        """
        a = ko.a; e = ko.e
        dm2_dt = 0.
        if e == 0.:
            r, v = ko.get_orbital_vectors(0.)
            dm2_dt = self.dm2_dt(hs, ko, r, v, opt)
        else:
            if  isinstance(a, (collections.Sequence, np.ndarray)):
                return np.array([self.dm2_dt_avg(hs, a_i, e, opt) for a_i in a])
            def integrand(phi):
                r, v = ko.get_orbital_vectors(phi)
                return self.dm2_dt(hs, ko, r, v, opt) / (1.+e*np.cos(phi))**2
            dm2_dt = (1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]

        return dm2_dt

    def __str__(self):
        return self.name


class DissipativeForceSS(DissipativeForce):
    """
    This is a model class from which the dissipative forces should be derived
    It assumes the dissipative force to just depend on the radius and total velocity of the secondary object
    And therefore not change the orbital plane
    """
    name = "DissipativeForceSS"
    m2_change = False


    def F(self, hs, ko, r, v, opt):
        """
        Placeholder function that models the dissipative force strength.

        Parameters:
            hs (HostSystem) : The object describing the properties of the host system
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            r  (float)      : The radius of the secondary (as in total distance to the MBH)
            v  (float)      : The total velocity
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The strength of the dissipative force, assumed to be antiparallel to the propagation direction
        """
        pass


    def dE_dt(self, hs, ko, opt):
        """
        The function calculates the energy loss due to a force F(r,v) by averaging over a Keplerian orbit
           with semimajor axis a and eccentricity e
        According to https://arxiv.org/abs/1908.10241

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy loss due to accretion
        """
        a = ko.a; e = ko.e
        if  isinstance(ko, (collections.Sequence, np.ndarray)):
            return np.array([self.dE_dt(hs, ko_i, opt) for ko_i in ko])
        if e == 0.:
            v = hs.omega_s(a)*a
            r, v = ko.get_orbital_parameters(0.)
            F = self.F(hs, ko, r, v, opt)
            F_proj = F*v
            return - F_proj
        else:
            def integrand(phi):
                r, v = ko.get_orbital_parameters(phi)
                F = self.F(hs, ko, r, v, opt)
                return F*v / (1.+e*np.cos(phi))**2
            return -(1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    def dL_dt(self, hs, ko, opt):
        """
        The function calculates the angular momentum loss due to a force F(r,v) by averaging over a Keplerian orbit
           with semimajor axis a and eccentricity e
        According to https://arxiv.org/abs/1908.10241

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum loss due to accretion
        """
        a = ko.a; e = ko.e
        def integrand(phi):
            r, v = ko.get_orbital_parameters(phi)
            F = self.F(hs, ko, r, v, opt)
            return F/v / (1.+e*np.cos(phi))**2
        return -(1.-e**2)**(3./2.)/2./np.pi *np.sqrt(ko.m_tot * a*(1.-e**2))* quad(integrand, 0., 2.*np.pi, limit = 100)[0]

    def dinclination_angle_dt(self, hs, ko, opt):
        """
        The function calculates the inclination agnle change due to a force F(r,v)
            calculated with osculating orbital elements

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum loss due to accretion
        """
        return 0.


    def dm2_dt(self, hs, ko, r, v, opt):
        """
        Placeholder function that models the mass gain/loss

        Parameters:
            hs (SystemProp) : The object describing the properties of the host system
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            r  (float)      : The position of the secondary
            v  (float)      : The velocity of the secondary
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The derivative of the secondary mass
        """
        return 0.


    def dm2_dt_avg(self, hs, ko, opt):
        """
        The function gives the mass gain due to accretion of the small black hole inside of a halo
           on a Keplerian orbit with semimajor axis a and eccentricity e
        For a circular orbit the dm2_dt function with the corresponding orbital velocity is used
            for an elliptic orbit the average of the expression is used

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The mass gain of the secondary
        """
        a = ko.a; e = ko.e
        dm2_dt = 0.
        if e == 0.:
            r, v = ko.get_orbital_parameters(0.)
            dm2_dt = self.dm2_dt(hs, ko, r, v, opt)
        else:
            if  isinstance(a, (collections.Sequence, np.ndarray)):
                return np.array([self.dm2_dt_avg(hs, a_i, e, opt) for a_i in a])
            def integrand(phi):
                r, v = ko.get_orbital_parameters(phi)
                return self.dm2_dt(hs, ko, r, v, opt) / (1.+e*np.cos(phi))**2
            dm2_dt = (1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]

        return dm2_dt

    def __str__(self):
        return self.name

class StochasticForce(DissipativeForce):
    """
    This class extends the dissipative force class
    to include a stochastic component that induces Brownian motion
    in the energy and angular momentum space

    See the Stochastic Solver for a thorough description
    """
    name = "StochasticForce"


    def dEdL_diffusion(self, hs, ko, opt):
        """
        Placeholder function that should return the Brownian motion in energy E and angular momentum L.
        The variances of E, L are expected on the diagonal [0,0], [1,1], and the covariance
         on the offdiagonal.

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : np.ndarray with shape=(2,2)
                The diffusion matrix for the SDE
        """
        pass



class GWLoss(DissipativeForceSS):
    name = "GWLoss"

    def dE_dt(self, hs, ko, opt):
        """
        The function gives the energy loss due to radiation of gravitational waves
            for a Keplerian orbit with semimajor axis a and eccentricity e
        According to Maggiore (2007)

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy loss due to radiation of gravitational waves of an Keplerian orbit
        """
        return -32./5. * ko.m_red**2 * ko.m_tot**3 / ko.a**5  / (1. - ko.e**2)**(7./2.) * (1. + 73./24. * ko.e**2 + 37./96. * ko.e**4)

    def dL_dt(self, hs, ko, opt):
        """
        The function gives the loss of angular momentum due to radiation of gravitational waves of the smaller object
           on a Keplerian orbit with semimajor axis a and eccentricity e
        According to Maggiore (2007)

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum loss due to radiation of gravitational waves
        """
        return -32./5. * ko.m_red**2 * ko.m_tot**(5./2.) / ko.a**(7./2.)  / (1. - ko.e**2)**2 * (1. + 7./8.*ko.e**2)


class ParameterizedForce(DissipativeForceSS):
    name = "ParameterizedForce"

    def __init__(self, alpha, beta, F_0=1.):
        self.alpha = alpha
        self.beta = beta
        self.F_0 = F_0

    def F(self, hs, ko, r, v, opt):
        """
        This function is a general parametrization as F ~ r^alpha v^beta

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            r  (float)      : The radius of the secondary (as in total distance to the MBH)
            v  (float)      : The total velocity
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the dynamical friction force - antiparralel to the velocity
        """
        F = self.F_0 * r**self.alpha * v**self.beta
        return F


class DynamicalFriction(DissipativeForceSS):
    name = "DynamicalFriction"

    def __init__(self, halo=None, ln_Lambda=-1, relativisticCorrections = False, haloPhaseSpaceDescription = False, includeHigherVelocities = True):
        self.halo = halo
        self.ln_Lambda = ln_Lambda
        self.relativisticCorrections = relativisticCorrections
        self.haloPhaseSpaceDescription = haloPhaseSpaceDescription
        self.includeHigherVelocities = haloPhaseSpaceDescription and includeHigherVelocities

    def F(self, hs, ko, r, v, opt):
        """
        The function gives the force of the dynamical friction of an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v
        The self.ln_Lambda is the Coulomb logarithm, for which different authors use different values. Set to -1 so that Lambda = sqrt(m1/m2)
        The self.relativisticCorrections parameter allows the use of the correction factor as given by eq (15) of
                https://arxiv.org/pdf/2204.12508.pdf ( except for the typo in the gamma factor )
        The self.useHaloPhaseSpaceDescription parameter allows to use not the total dark matter density at r, but uses the halo phase space description
            such that only particles below a given v_max scatter. This option requires sp.halo to be of type DynamicSS.
            v_max can be provided via self.v_max. If v_max is None, it is taken to be the orbital velocity.

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            r  (float)      : The radius of the secondary (as in total distance to the MBH)
            v  (float)      : The total velocity
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the dynamical friction force - antiparralel to the velocity
        """
        ln_Lambda = self.ln_Lambda
        halo = self.halo or hs.halo
        v_rel = v

        if ln_Lambda < 0.:
            ln_Lambda = np.log(ko.m1/ko.m2)/2.

        relCovFactor = 1.
        if self.relativisticCorrections:
            relCovFactor = (1. + v_rel**2)**2 / (1. - v_rel**2)

        density = halo.density(r)
        bracket = ln_Lambda
        if self.haloPhaseSpaceDescription:
            alpha, beta, delta  = 0., 0., 0.

            v2_list = np.linspace(0., v_rel**2, 3000)
            f_list = halo.f(np.clip(halo.potential(r) - 0.5*v2_list, 0.,None))
            alpha =  4.*np.pi*simpson(v2_list * f_list, x=np.sqrt(v2_list)) / density # TODO: Change Normalization of f from density to 1?

            while self.includeHigherVelocities:
                v_esc = np.sqrt(2*halo.potential(r))
                if np.abs(v_esc - v_rel)/v_esc < 2e-6 or v_rel > v_esc:
                    break
                v_list = np.linspace(v_rel+1e-7*v_rel, v_esc, 3000)
                f_list = halo.f(np.clip(halo.potential(r) - 0.5*v_list**2, 0., None))

                beta =  4.*np.pi*simpson(v_list**2 * f_list * np.log((v_list + v_rel)/(v_list-v_rel)), x=v_list) / density
                beta = np.nan_to_num(beta) # in case v_list ~ v_rel
                delta =  -8.*np.pi*v_rel*simpson(v_list * f_list, x=v_list) / density
                break
            bracket = (alpha * ln_Lambda + beta + delta)
            #print(rf"r={r:.3e}({r/sp.r_isco():.3e} r_isco), v={v_rel:.3e}, alpha={alpha:.3e}, beta={beta:.3e}, delta={delta:.3e}, bracket={bracket:.3e}")

        F_df = 4.*np.pi * relCovFactor * ko.m2**2 * density * bracket / v_rel**2
        F_df = np.nan_to_num(F_df)
        return F_df

    def dinclination_angle_dt(self, hs, ko, opt):
        """
        For now we assume the (DM) distribution to be spherically symmetric, so there is no inclination change
        """
        return 0.


class GasDynamicalFriction(DissipativeForce):
    name = "GasDynamicalFriction"

    def __init__(self, disk = None, ln_Lambda= -1., relativisticCorrections = False, frictionModel='Ostriker'):
        """
        Constructor for the GasDynamicalFriction class

        ln_Lambda :  the Coulomb logarithm, for which different authors use different values. Set to -1 so that Lambda = sqrt(m1/m2)
        frictionModel :
            'Ostriker' refers to eq (3) of https://arxiv.org/pdf/2203.01334.pdf
            'Sanchez-Salcedo' refers to eq (5) of https://arxiv.org/pdf/2006.10206.pdf
        """
        self.ln_Lambda = ln_Lambda
        if not frictionModel in ['Ostriker', 'Sanchez-Salcedo']:
            raise Exception(f"Gas dynamical friction model not recognized: {frictionModel}")
        self.frictionModel = frictionModel
        self.disk = disk
        self.relativisticCorrections = relativisticCorrections

    def F(self, hs, ko, pos, v, opt):
        """
        The function gives the force of the dynamical friction of an object inside a gaseous disk at radius r
            and with velocity v

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            pos  (np.ndarray) : The position of the secondary in the XYZ fundamental plane
            v  (np.ndarray) : The velocity vector of the secondary in the XYZ fundamental plane
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the dynamical friction force
        """
        ln_Lambda = self.ln_Lambda
        disk = self.disk or hs.halo
        r, phi, z = kepler.KeplerOrbit.from_xy_plane_to_rhophi_plane(pos)

        v_gas = disk.velocity(r, phi, z=z) #  TODO: Improve
        v_rel = ( v - v_gas if opt.considerRelativeVelocities
                        else v )
        v_rel_tot = np.sqrt(np.sum(v_rel*v_rel))
        # print(v, v_gas, v_rel)
        relCovFactor = 1.
        if self.relativisticCorrections:
            relCovFactor = (1. + v_rel_tot**2)**2 / (1. - v_rel_tot**2)

        if ln_Lambda < 0.:
            ln_Lambda = np.log(ko.m1/ko.m2)/2.

        if self.frictionModel == 'Ostriker':
            c_s = disk.soundspeed(r)
            I = np.where( v_rel_tot >= c_s,
                                1./2. * np.log(1. - (c_s/v_rel_tot)**2) + ln_Lambda, # supersonic regime
                                1./2. * np.log((1. + v_rel_tot/c_s)/(1. - v_rel_tot/c_s)) - v_rel_tot/c_s) # subsonic regime
            ln_Lambda = I
        elif self.frictionModel == 'Sanchez-Salcedo':
                H = disk.scale_height(r)
                R_acc = 2.*ko.m2 /v_rel_tot**2
                ln_Lambda =  7.15*H/R_acc

        F_df = 4.*np.pi * ko.m2**2 * relCovFactor * disk.density(r, z=z) * ln_Lambda / v_rel_tot**2
        #print(v, v_gas, v_rel, F_df)
        F_df = np.nan_to_num(F_df)
        return F_df* v_rel / v_rel_tot


class GasGeometricDrag(DissipativeForce):
    name = "GasDynamicalFriction"

    def __init__(self, r_stellar, disk = None, C_drag= 1., relativisticCorrections = False):
        """
        Constructor for the GasDynamicalFriction class

        """
        self.r_stellar = r_stellar
        self.disk = disk
        self.C_drag = C_drag
        self.relativisticCorrections = relativisticCorrections

    def F(self, hs, ko, pos, v, opt):
        """
        The function gives the force of the dynamical friction of an object inside a gaseous disk at radius r
            and with velocity v

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            pos  (np.ndarray) : The position of the secondary in the XYZ fundamental plane
            v  (np.ndarray) : The velocity vector of the secondary in the XYZ fundamental plane
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the dynamical friction force
        """
        disk = self.disk or hs.halo
        r, phi, z = kepler.KeplerOrbit.from_xy_plane_to_rhophi_plane(pos)

        v_gas = disk.velocity(r, phi, z=z) #  TODO: Improve
        v_rel = ( v - v_gas if opt.considerRelativeVelocities
                        else v )
        v_rel_tot = np.sqrt(np.sum(v_rel*v_rel))

        relCovFactor = 1.
        if self.relativisticCorrections:
            relCovFactor = (1. + v_rel_tot**2)**2 / (1. - v_rel_tot**2)


        F_gd = self.C_drag / 2. * 4.*np.pi * self.r_stellar**2 * disk.density(r,z=z) * v_rel_tot**2
        #print(v, v_gas, v_rel, F_df)
        F_gd = np.nan_to_num(F_gd)
        return F_gd* v_rel / v_rel_tot

class AccretionLoss(DissipativeForceSS):
    name = "AccretionLoss"
    m2_change = True

    def __init__(self, halo = None, accretionModel = 'Collisionless', withSoundspeed = False, includeRecoil=False):
        """
        Constructor for the AccretionLoss function

        accretionModel :
            'Collisionless' (default): according to https://arxiv.org/pdf/1711.09706.pdf
            'Bondi-Hoyle' : according to https://arxiv.org/pdf/1302.2646.pdf
        withSoundspeed : Whether to include the soundspeed in the Bondi-Hoyle model
        includeRecoil  : Whether to include recoil effects from the collisions
        """
        self.halo = halo
        if not accretionModel in ['Collisionless', 'Bondi-Hoyle']:
            raise Exception(f"Accretion model not recognized: {accretionModel}")
        self.accretionModel = accretionModel
        self.withSoundspeed = withSoundspeed
        self.includeRecoil = includeRecoil

    def BH_cross_section(self, hs, ko, r, v_rel, opt):
        """
        The function gives the cross section of a small black hole (m2) moving through a halo of particles
        Choose model through opt.accretionModel parameter
            'Collisionless' (default): according to https://arxiv.org/pdf/1711.09706.pdf
            'Bondi-Hoyle' : according to https://arxiv.org/pdf/1302.2646.pdf

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            r  (float)      : The radial position of the black hole in the halo
            v  (float)      : The speed of the secondary
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The black hole cross section
        """
        halo = self.halo or hs.halo
        if self.accretionModel == 'Bondi-Hoyle':

            dm_soundspeed2 = halo.soundspeed(r)**2 if self.withSoundspeed else 0.

            return 4.*np.pi * ko.m2**2 / (v_rel**2 +  dm_soundspeed2)**(3./2.)  / v_rel

        elif self.accretionModel == 'Collisionless':
            return (np.pi * ko.m2**2. / v_rel**2.) * (8. * (1. - v_rel**2.))**3 / (4. * (1. - 4. * v_rel**2. + (1. + 8. * v_rel**2.)**(1./2.)) * (3. - (1. + 8. * v_rel**2.)**(1./2.))**2.)
            #return 16. * np.pi * sp.m2**2 / v**2  * (1. + v**2)


    def dm2_dt(self, hs, ko, r, v, opt):
        """
        The function gives the mass gain due to accretion of the small black hole inside of the dark matter halo
           for a small black hole with relative velocity v to the halo at radius r
        The equation of https://arxiv.org/pdf/1711.09706.pdf is used

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            r  (float)      : The radius of the secondary (as in total distance to the MBH)
            v  (float)      : The total velocity
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The mass gain due to accretion
        """
        halo = self.halo or hs.halo
        return halo.density(r) * v * self.BH_cross_section(hs, ko, r, v, opt)


    def F(self, hs, ko, r, v, opt):
        """
        The function gives the total force acting on an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v through accretion effects

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            r  (float)      : The radius of the secondary (as in total distance to the MBH)
            v  (float)      : The total velocity
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the accretion force
        """
        F_acc = self.F_acc(hs, ko, r, v, opt)
        F_acc_rec = self.F_acc_recoil(hs, ko, r, v, opt) if self.includeRecoil else 0.
        return F_acc + F_acc_rec

    def F_acc(self, hs, ko, r, v, opt):
        """
        The function gives the force mimicing the accretion effects of an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            r  (float)      : The radius of the secondary (as in total distance to the MBH)
            v  (float)      : The total velocity
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the accretion force
        """
        return self.dm2_dt(hs, ko, r, v, opt) *  v


    def F_recoil(self, hs, ko, r, v, opt):
        """
        The function gives the recoil force of the accretion of an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            r  (float)      : The radius of the orbiting object
            v  (float or tuple)   : The speed of the orbiting object, either as |v|, or (v_r, v_theta) split into the direction of r, theta
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the accretion recoil force
        """
        return self.dm2_dt(hs, ko, r, v, opt) *  v



class GasInteraction(DissipativeForceSS):
    name = "GasInteraction"

    def __init__(self, disk = None, gasInteraction = 'gasTorqueLossTypeI', alpha=0.1, fudgeFactor=1.):
        if not gasInteraction in ['gasTorqueLossTypeI', 'gasTorqueLossTypeII']:
            raise Exception(f"Gas Interaction type not recognized: {gasInteraction}")
        self.gasInteraction = gasInteraction
        self.alpha = None
        self.fudgeFactor = fudgeFactor
        self.disk = disk

    def F(self, hs, ko, r, v, opt):
        """
        The function gives the force an accretion disk would exert on a smaller black hole for different models of gas interaction

        Choose model through opt.gasInteraction parameter
            'gasTorqueLoss' : according to https://arxiv.org/pdf/2206.05292.pdf
            'dynamicalFriction' : according to https://arxiv.org/pdf/2006.10206.pdf

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            r  (float)      : The radius of the secondary (as in total distance to the MBH)
            v  (float)      : The total velocity
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the force through gas interactions
        """
        disk = self.disk or hs.halo
        #v_gas = disk.velocity(r)
        #v_rel = ( self.get_relative_velocity(v, v_gas)[0] if opt.considerRelativeVelocities
        #                else self.get_relative_velocity(v, 0.)[0] )

        if self.gasInteraction == 'gasTorqueLossTypeI':
            mach_number = disk.mach_number(r)
            Sigma = disk.surface_density(r)
            Omega = hs.omega_s(r)
            Gamma_lin = Sigma*r**4 * Omega**2 * (ko.m2/ko.m1)**2 * mach_number**2

            F_gas = Gamma_lin * ko.m2/ko.m1 / r

        elif self.gasInteraction == 'gasTorqueLossTypeII':
            mach_number = disk.mach_number(r)
            Sigma = disk.surface_density(r)
            alpha = disk.alpha if hasattr(disk, 'alpha') else self.alpha # requires ShakuraSunyaevDisk atm

            Omega = hs.omega_s(r)
            Gamma_vis = 3.*np.pi * alpha * Sigma * r**4 * Omega**2 / mach_number**2 if mach_number > 0. else 0.

            fudge_factor = self.fudgeFactor
            Gamma_gas = fudge_factor * Gamma_vis

            F_gas = Gamma_gas * ko.m2/ko.m1 / r

        '''
        elif self.gasInteraction == 'gasTorqueLossTypeITanaka':
            C = 7.2e-10
            n_r = 8.
            n_alpha = -1.
            n_fedd = -3.
            n_M1 = 1.
            A = (C * (hs.halo.alpha/0.1)**n_alpha * (hs.halo.f_edd/hs.halo.eps)**n_fedd
                    * (hs.m1/1e6/c.solar_mass_to_pc)**n_M1 )
            L0 = 32./5. * ko.m2 / hs.m1 * (r/hs.m1)**(-7./2.)
            L_disk = A * (r/10./hs.m1)**n_r * L0

            F_gas = L_disk * ko.m2/ko.m1 / r

        elif self.gasInteraction == 'hydrodynamicDragForce':
            C_d = 1.
            R_p = 2.*ko.m2
            rho = hs.halo.density(r)

            F_gas = 1./2. * C_d * np.pi * rho * v_rel**2 * R_p**2
        '''

        return F_gas

class StellarDiffusion(StochasticForce):
    """
    The class for modeling stellar diffusion effects in an IMRI. There is an averaged loss and a stochastic component to this
    This is modeled after https://arxiv.org/pdf/2304.13062.pdf

    Attributes:
        stellarDistribution : imripy.halo.MatterHaloDF
            Object describing the stellar distribution
        E_m_s : float
            The first mass moment of the stellar distribution
        E_m_s2 : float
            The second mass moment of the stellar distribution
        CoulombLogarithm : float
            The CoulombLogarithm describing the strength of the scattering
        m : float
            The mass of the secondary object subject to stellar diffusion

    """
    name = "Stellar Diffusion"

    def __init__(self, stellarDistribution : imripy.halo.MatterHaloDF, m1, m2, E_m_s = c.solar_mass_to_pc, E_m_s2 = c.solar_mass_to_pc**2, CoulombLogarithm=None):
        """
        The constructor for the StellarDiffusion class
        The values are initialized according to https://arxiv.org/pdf/2304.13062.pdf if not provided

        Parameters:
            istellarDistribution : imripy.halo.MatterHaloDF
                Object describing the stellar distribution
            m1 : float
                The mass of the central MBH
            m2 : float
                The mass of the secondary
            E_m_s : (optional) float
                The first mass moment of the stellar distribution
            E_m_s2 :(optional)  float
                The second mass moment of the stellar distribution
            CoulombLogarithm : (optional) float
                The CoulombLogarithm describing the strength of the scattering - alternatively m1 is used for estimation
        """
        super().__init__()
        self.stellarDistribution = stellarDistribution

        self.E_m_s = E_m_s
        self.E_m_s2 = E_m_s2
        self.m = m2
        self.CoulombLogarithm = CoulombLogarithm or np.log(0.4 * m1 / self.E_m_s)

        self.calc_velocity_diffusion_coeff(m1)



    def calc_velocity_diffusion_coeff(self, m1, acc=1e-8):
        """
        Calculates the velocity diffusion coefficients and saves them in the class for later use as a lambda function.
        This should only be needed once (or when the distribution function stellarDistribution.f changes)
        Eq (24)-(28) from https://arxiv.org/pdf/2304.13062.pdf
        """

        r_grid = np.geomspace(2.*m1, 1e8*2*m1, 60)
        v_grid = np.geomspace( np.sqrt(2.* self.stellarDistribution.potential(r_grid[-1]))/1e4, np.sqrt(2.*self.stellarDistribution.potential(r_grid[0]))*1e4, 101)
        R_grid, V_grid = np.meshgrid(r_grid, v_grid)

        # The distribution function depends on the specific energy Eps
        f = lambda v, r: self.stellarDistribution.f( np.clip(- v**2 /2. + self.stellarDistribution.potential(r), 0., None)) / self.E_m_s
        #f = lambda v, r: self.stellarDistribution.f( - v**2 /2. + self.stellarDistribution.potential(r))

        # Calculte E_1, F_2, and F_4 for interpolation
        E_1_int = np.zeros(np.shape(V_grid))
        #E_1_int_alt = np.zeros(np.shape(V_grid))
        for i, r in enumerate(r_grid):
            #E_1_int_alt[:,i] = np.array([quad(lambda v_int : v_int * f(v_int, r), v, np.sqrt(2.*self.stellarDistribution.potential(r)))[0] for v in v_grid])
            #E_1_int_alt[:,i] = odeint( lambda a, v_int : -v_int *f(-v_int, r),  0, -v_grid[::-1], atol=acc)[:,0][::-1]
            for j,v in enumerate(v_grid):
                if v > np.sqrt(2.*self.stellarDistribution.potential(r)):
                    continue
                v_int_grid = np.linspace(v, np.sqrt(2.*self.stellarDistribution.potential(r)), 1000)
                E_1_int[j,i] = simpson(v_int_grid*f(v_int_grid, r), x=v_int_grid)
        E_1_int /=  V_grid
        #E_1_int_alt /= V_grid
        E_1 = CloughTocher2DInterpolator( list(zip(np.log(R_grid).flatten(), np.log(V_grid).flatten())),
                                             E_1_int.flatten())

        F_2_int = np.zeros(np.shape(V_grid))
        #F_2_int_alt = np.zeros(np.shape(V_grid))
        for i, r in enumerate(r_grid):
            #F_2_int_alt[:,i] = np.array([quad(lambda v_int : v_int**2 * f(v_int, r), 0, v)[0] for v in v_grid])
            #F_2_int_alt[:,i] = odeint( lambda a, v_int : v_int**2 * f(np.exp(v_int), r), 0., v_grid)[:,0]
            for j,v in enumerate(v_grid):
                v_int_grid = np.linspace(0, v, 1000)
                F_2_int[j,i] = simpson(v_int_grid**2 *f(v_int_grid, r), x=v_int_grid)
        F_2_int /= V_grid**2
        #F_2_int_alt /= V_grid**2
        F_2 = CloughTocher2DInterpolator(list(zip(np.log(R_grid).flatten(), np.log(V_grid).flatten())),
                                            F_2_int.flatten() )

        F_4_int = np.zeros(np.shape(V_grid))
        #F_4_int_alt = np.zeros(np.shape(V_grid))
        for i, r in enumerate(r_grid):
            #F_4_int_alt[:,i] = np.array([quad(lambda v_int : v_int**4 * f(v_int, r), 0, v)[0] for v in v_grid])
            #F_4_int_alt[:,i] = odeint( lambda a, v_int : v_int**4 * f(v_int, r), 0., v_grid, atol=acc)[:,0]
            for j,v in enumerate(v_grid):
                v_int_grid = np.linspace(0, v, 1000)
                F_4_int[j,i] = simpson(v_int_grid**4 *f(v_int_grid, r), x=v_int_grid)
        F_4_int /= V_grid**4
        #F_4_int_alt /= V_grid**4
        F_4 = CloughTocher2DInterpolator(list(zip(np.log(R_grid).flatten(), np.log(V_grid).flatten())),
                                            F_4_int.flatten() )
        '''
        plt.figure(figsize=(12,8))
        n1_2 = int(len(r_grid)/2)
        plt.loglog(v_grid, np.array([f(v, r_grid[0]) for v in v_grid]), label='f(r_0)')
        plt.loglog(v_grid, np.array([f(v, r_grid[n1_2]) for v in v_grid]), label='f(r_1/2)')
        plt.loglog(v_grid, np.array([f(v, r_grid[-1]) for v in v_grid]), label='f(r_1)')

        #plt.loglog(v_grid, F_2_int[:,0], label='F_2(r_0)', linestyle='-.')
        #plt.loglog(v_grid, F_2_int[:,n1_2], label='F_2(r_1/2)', linestyle='-.')
        #plt.loglog(v_grid, F_2_int[:,-1], label='F_2(r_1)', linestyle='-.')
        #plt.loglog(v_grid, F_2_int_alt[:,0], label='F_2(r_0)', linestyle='--')
        #plt.loglog(v_grid, F_2_int_alt[:,n1_2], label='F_2(r_1/2)', linestyle='--')
        #plt.loglog(v_grid, F_2_int_alt[:,-1], label='F_2(r_1)', linestyle='--')

        #plt.loglog(v_grid, F_4_int[:,0], label='F_4(r_0)', linestyle='-.')
        #plt.loglog(v_grid, F_4_int[:,n1_2], label='F_4(r_1/2)', linestyle='-.')
        #plt.loglog(v_grid, F_4_int[:,-1], label='F_4(r_1)', linestyle='-.')
        #plt.loglog(v_grid, F_4_int_alt[:,0], label='F_4(r_0)', linestyle='--')
        #plt.loglog(v_grid, F_4_int_alt[:,n1_2], label='F_4(r_1/2)', linestyle='--')
        #plt.loglog(v_grid, F_4_int_alt[:,-1], label='F_4(r_1)', linestyle='--')

        #plt.loglog(v_grid, E_1_int[:,0], label='E_1(r_0)', linestyle='-.')
        #plt.loglog(v_grid, E_1_int[:,n1_2], label='E_1(r_1/2)', linestyle='-.')
        #plt.loglog(v_grid, E_1_int[:,-1], label='E_1(r_1)', linestyle='-.')
        #plt.loglog(v_grid, E_1_int_alt[:,0], label='E_1(r_0)', linestyle='--')
        #plt.loglog(v_grid, E_1_int_alt[:,n1_2], label='E_1(r_1/2)', linestyle='--')
        #plt.loglog(v_grid, E_1_int_alt[:,-1], label='E_1(r_1)', linestyle='--')
        plt.legend(); plt.grid()
        '''

        # Make lambda functions for the velocity diffusion coefficients
        E_v_par_pref =  -16.*np.pi**2 *(self.E_m_s2 + self.m * self.E_m_s ) * self.CoulombLogarithm
        self.E_v_par = lambda r, v: E_v_par_pref * F_2(np.log(r), np.log(v))

        E_v_par2_pref = 32./3.*np.pi**2 * self.E_m_s2 * self.CoulombLogarithm
        self.E_v_par2 = lambda r, v: E_v_par2_pref * v * (F_4(np.log(r), np.log(v)) + E_1(np.log(r), np.log(v)))

        E_v_ort2_pref = E_v_par2_pref
        self.E_v_ort2 = lambda r, v: E_v_ort2_pref * v * (3*F_2(np.log(r), np.log(v)) - F_4(np.log(r), np.log(v)) + 2*E_1(np.log(r), np.log(v)))

        plt.figure(figsize=(12,8))

        n1_2 = int(len(r_grid)/2)
        plt.loglog(v_grid, np.abs(self.E_v_par(r_grid[n1_2], v_grid)), label='<$Delta v_\parallel>(r_1/2)$')
        plt.loglog(v_grid, 4.*np.pi *(self.E_m_s + self.m ) * self.CoulombLogarithm* np.array([imripy.halo.MatterHaloDF.density(self.stellarDistribution, r_grid[n1_2], v_max=v)/v**2  for v in v_grid]),
                    label='distr(r_1/2)', linestyle='--')
        plt.loglog(v_grid, np.abs(self.E_v_par2(r_grid[n1_2], v_grid)), label='$<Delta v_\parallel^2>(r_1/2)$')
        plt.loglog(v_grid, np.abs(self.E_v_ort2(r_grid[n1_2], v_grid)), label='$<Delta v_\perp^2>(r_1/2)$')
        Clog_prime = self.CoulombLogarithm
        sigma_f = np.sqrt(m1 / (1. + self.stellarDistribution.alpha) / r_grid[n1_2])
        rho_f = self.stellarDistribution.density(r_grid[n1_2])
        C = 8.* np.sqrt(2.*np.pi)/3. * self.E_m_s * rho_f / sigma_f * Clog_prime
        plt.axhline(C, linestyle='--')

        plt.loglog(v_grid, np.abs(self.E_v_par(r_grid[0], v_grid)), label='$<Delta v||>(r_0)$')
        plt.loglog(v_grid, 4.*np.pi *(self.E_m_s + self.m ) * self.CoulombLogarithm* np.array([imripy.halo.MatterHaloDF.density(self.stellarDistribution, r_grid[0], v_max=v)/v**2  for v in v_grid]),
                    label='distr(r_0)', linestyle='--')
        plt.loglog(v_grid, np.abs(self.E_v_par2(r_grid[0], v_grid)), label='$<Delta v_\parallel^2>(r_0)$')
        plt.loglog(v_grid, np.abs(self.E_v_ort2(r_grid[0], v_grid)), label='$<Delta v_\perp^2>(r_0)$')
        Clog_prime = self.CoulombLogarithm
        sigma_f = np.sqrt(m1 / (1. + self.stellarDistribution.alpha) / r_grid[0])
        rho_f = self.stellarDistribution.density(r_grid[0])
        C = 8.* np.sqrt(2.*np.pi)/3. * self.E_m_s * rho_f / sigma_f * Clog_prime
        plt.axhline(C, linestyle='--')

        plt.legend(); plt.grid()



    def dE_dt(self, hs, ko, opt):
        """
        Calculates the average energy loss due to stellar diffusion.
        Eq (14) from https://arxiv.org/pdf/2304.13062.pdf

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy loss due to stellar diffusion
        """
        a = ko.a; e = ko.e
        def integrand(phi):
            r, v = ko.get_orbital_parameters(phi)
            deps = v * self.E_v_par(r, v) + self.E_v_par2(r, v) / 2. + self.E_v_ort2(r, v) / 2. # the negative signs cancel out ?
            return deps / (1.+e*np.cos(phi))**2
        dE_dt = self.m * (1.-e**2)**(3./2.) /2./np.pi * quad(integrand, 0., 2.*np.pi, limit=100)[0] # the 1/T is in the 1/2pi
        return - dE_dt

    def dL_dt(self, hs, ko, opt):
        """
        Calculates the average angular momentum loss due to stellar diffusion.
        Eq (16) from https://arxiv.org/pdf/2304.13062.pdf

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum loss due to stellar diffusion
        """
        a = ko.a; e = ko.e
        J = np.sqrt(ko.m1 * a * (1.-e**2))
        def integrand(phi):
            r, v = ko.get_orbital_parameters(phi)
            dJ = J / v * self.E_v_par(r, v) + r**2 / J / 4. * self.E_v_ort2(r, v)
            return dJ / (1.+e*np.cos(phi))**2
        dL_dt = self.m  * (1.-e**2)**(3./2.) /2./np.pi * quad(integrand, 0., 2.*np.pi, limit=100)[0]  # the 1/T is in the 1/2pi
        #plt.plot(np.linspace(0., 2.*np.pi, 100), [integrand(p) for p in np.linspace(0., 2.*np.pi, 100)])
        return - dL_dt

    def dinclination_angle_dt(self, hs, ko, opt):
        """
        For now we assume the stellar distribution to be spherically symmetric, so there is no inclination change
        """
        return 0.

    def dEdL_diffusion(self, hs, ko, opt):
        """
        Calculates the matrix for the diffusion term of the SDE due to stellar diffusion.
        The variances are on the diagonal according to Eq(15) and (17), the covariance is on the off-diagonal Eq(18)

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : np.matrix
                The diffusion matrix
        """
        a = ko.a; e = ko.e
        J = np.sqrt(ko.m1 * a * (1.-e**2))
        T = 2.*np.pi * np.sqrt(a**3 / ko.m1)

        def integrand_deps2(phi):
            r, v = ko.get_orbital_parameters(phi)
            deps2 = v**2 *  self.E_v_par2(r, v)
            return deps2 / (1.+e*np.cos(phi))**2
        D_EE = 2.* self.m**2 * (1.-e**2)**(3./2.)  /2./np.pi * quad(integrand_deps2, 0., np.pi, limit=60)[0] # The factor of 2 because we only integrate over half the orbit pi

        def integrand_dJ2(phi):
            r, v = ko.get_orbital_parameters(phi)
            dJ2 = J**2 / v**2 * self.E_v_par2(r, v) + 1./2. *( r**2 - J**2/v**2) * self.E_v_ort2(r, v)
            return dJ2 / (1.+e*np.cos(phi))**2
        D_LL = 2.* self.m**2 * (1.-e**2)**(3./2.) /2./np.pi * quad(integrand_dJ2, 0., np.pi, limit=60)[0]

        def integrand_dJdeps(phi):
            r, v = ko.get_orbital_parameters(phi)
            dJdeps = J *self.E_v_par2(r, v)
            return dJdeps / (1.+e*np.cos(phi))**2
        D_EL =  2.* self.m**2 * (1.-e**2)**(3./2.) /2./np.pi * quad(integrand_dJdeps, 0., np.pi, limit=60)[0]

        sigma = np.zeros((2,2))
        sigma[0,0] =  np.sqrt(D_EE)
        sigma[1,0] =  D_EL / np.sqrt(D_EE)
        sigma[1,1] =  np.sqrt( D_LL - D_EL**2 / D_EE)

        #print(rf"D_EE = {D_EE:.3e}, D_LL = {D_LL:.3e}, D_EL = {D_EL:.3e}, 2D={np.matmul(sigma, sigma.T)}")

        return sigma



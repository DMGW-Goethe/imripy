import numpy as np
from scipy.integrate import quad, quad_vec, odeint, simpson
from scipy.interpolate import interp1d, LinearNDInterpolator
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
            e_phi = ko.get_orbital_decomposition_in_fundamental_xy_plane(phi)[1]
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
            e_z = ko.get_orbital_decomposition_in_fundamental_xy_plane(phi)[2]
            W = np.sum(F*e_z) / ko.m2
            di_dt = np.sqrt(np.sum(r*r)) * np.cos(ko.periapse_angle + phi) * W / Omega / a**2 / np.sqrt(1-e**2)
            return  di_dt / (1.+e*np.cos(phi))**2
        return -(1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    def dm2_dt(self, hs, ko, r, v, opt):
        """
        Placeholder function that models the mass gain/loss throughout the orbit

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
        The function gives the mass gain due to accretion of the secondary as an average over an orbit
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

        v_gas = disk.velocity(r, phi, z=z)
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


class StochasticForce(DissipativeForce):
    """
    This class extends the dissipative force class
    to include a stochastic component that induces Brownian motion
    in the energy and angular momentum space

    See the Stochastic Solver for a thorough description
    """
    name = "StochasticForce"

    def da_dt(self, Classic, hs, ko, opt):
        """
        Modified da_dt that takes into account the Stochastic contribution via the Ito formula
            in the time derivative of a
        This is called in the inspiral.Classic case, but not in the inspiral.Stochastic case, to avoid recomputing too many objects

        Parameters
        -------
            Classic (imripy.inspiral.Classic) : The class to avoid a circular import :'(
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns
        -------
            da_dt : float
                The time derivative of a containing the stochastic contribution
        """
        a = ko.a; e = ko.e
        E = Classic.E_orbit(hs, ko, opt)

        # Classic contribution
        dE_dt = self.dE_dt(hs, ko, opt)
        dE_orbit_da = Classic.dE_orbit_da(hs, ko, opt)
        da_dt_classic = dE_dt / dE_orbit_da

        # Stochastic contribution
        if hasattr(self, "D_EE"):
            D_EE = self.D_EE(hs, ko)
        else:
            sigma = self.dEdL_diffusion(hs, ko, opt=opt)
            D = np.matmul(sigma, sigma.T)
            D_EE = D[0,0]
        H_a = -2.*a/E**2
        da_dt_stochastic =  D_EE*H_a / 2.

        return (da_dt_classic + da_dt_stochastic)


    def de_dt(self, Classic, hs, ko, opt):
        """
        Modified de_dt that takes into account the stochastic contribution via the Ito formula
            in the time derivative of e
        This is called in the inspiral.Classic case, but not in the inspiral.Stochastic case, to avoid recomputing too many objects

        Parameters
        -------
            Classic (imripy.inspiral.Classic) : The class to avoid a circular import :'(
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns
        -------
            da_dt : float
                The time derivative of a containing the stochastic contribution
        """

        a = ko.a; e = ko.e
        E = Classic.E_orbit(hs, ko, opt)
        L = Classic.L_orbit(hs, ko, opt)

        # Classic contribution
        dE_dt = self.dE_dt(hs, ko, opt)
        dL_dt = self.dL_dt(hs, ko, opt)
        de_dt_classic =  - (1.-ko.e**2)/2./ko.e *(  dE_dt/E + 2. * dL_dt/L   )

        # Stochastic contribution
        sigma = self.dEdL_diffusion(hs, ko, opt=opt)
        D = np.matmul(sigma, sigma.T)

        pref = 2./ko.m_red**3 / ko.m_tot**2
        He = -pref * np.array([[ - pref * L**4 / 4. / e**3, L * (e**2 + 1.) / 2. / e**3],
                                [0.,  E / e**3]])
        He[1,0] = He[0,1]
        de_dt_stochastic =  np.sum(D*He)/2. # elementwise addition

        return (de_dt_classic + de_dt_stochastic)



    def dEdL_diffusion(self, hs, ko, opt):
        """
        Function that returns the Brownian motion in energy E and angular momentum L,
            calculated with the help of the diffusion coefficients the class defined
            D_EE, D_EL, D_LL
        If these functions are not defined this function has to be overwritten!

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : np.ndarray with shape=(2,2)
                The diffusion matrix for the SDE
        """
        D_EE = self.D_EE(hs, ko)
        D_EL = self.D_EL(hs, ko)
        D_LL = self.D_LL(hs, ko)

        sigma = np.zeros((2,2))
        sigma[0,0] =  np.sqrt(D_EE)
        sigma[1,0] =  D_EL / np.sqrt(D_EE)                  if np.abs(D_EE) > 0. else 0.
        sigma[1,1] =  np.sqrt( D_LL - D_EL**2 / D_EE )       if np.abs(D_EE) > 0. else np.sqrt(D_LL)

        #print(rf"D_EE = {D_EE:.3e}, D_LL = {D_LL:.3e}, D_EL = {D_EL:.3e}, D={np.matmul(sigma, sigma.T)}")
        return sigma

    def dinclination_angle_dt(self, hs, ko, opt):
        """
        For now we assume the simple spherically symmetric case with no inclination change. Here, the formalism could be expanded
        """
        return 0.


class StellarDiffusion(StochasticForce):
    """
    The class for modeling stellar diffusion effects a la https://arxiv.org/pdf/1508.01390.pdf

    Attributes:
        stellarDistribution : imripy.halo.MatterHaloDF
            Object describing the stellar distribution
        E_m_s : float
            The first mass moment of the stellar distribution
        CoulombLogarithm : float
            The CoulombLogarithm describing the strength of the scattering

    """
    name = "Stellar Diffusion"

    def __init__(self, hs, stellarDistribution : imripy.halo.MatterHaloDF, E_m_s = c.solar_mass_to_pc, CoulombLogarithm=None):
        """
        The constructor for the StellarDiffusion class
        The values are initialized according to https://arxiv.org/pdf/2304.13062.pdf if not provided

        Parameters:
            hs : imripy.merger_system.HostSystem
                Host System Object
            stellarDistribution : imripy.halo.MatterHaloDF
                Object describing the stellar distribution
            m1 : float
                The mass of the central MBH
            E_m_s : (optional) float
                The first mass moment of the stellar distribution
            CoulombLogarithm : (optional) float
                The CoulombLogarithm describing the strength of the scattering - alternatively m1 is used for estimation
        """
        super().__init__()
        self.stellarDistribution = stellarDistribution
        self.E_m_s = E_m_s
        self.CoulombLogarithm = CoulombLogarithm or np.log(0.4 * hs.m1 / self.E_m_s)

        self.calc_velocity_diffusion_coeff(hs.m1)

    def f(self, E):
        """
        Brings the distribution function from the stellarDistribution class to the right normalization to be compatible with
            https://arxiv.org/pdf/1508.01390.pdf
        In the code (and with previous DM literature), the distribution function is normalized to the mass, while in the stellar diffusion literature
            it is normalized to the number density -> Normalize by a factor of the average mass

        Parameters:
            E (float) : The specific energy ( E = m1/2a )

        Returns:
            f (float) : The value of the distribution function
        """
        return self.stellarDistribution.f(E) / self.E_m_s

    def potential(self, r):
        """
        The potential of the spherically symmetric system

        Parameters:
            r (float) : The radius

        Returns:
            phi (float) : The value of the potential
        """
        return self.stellarDistribution.potential(r)

    def calc_velocity_diffusion_coeff(self, m1, right_r=1e8, n_r = 200, n_v = 301):
        """
        Calculates the velocity diffusion coefficients and saves them in the class for later use as a lambda function.
        This should only be needed once (or when the distribution function stellarDistribution.f changes)
        See App. of https://arxiv.org/pdf/2304.13062.pdf

        Parameters:
            right_r (float): The right end of the grid in r, multiplying r_isco
            n_r (int) : The size of the r grid
            n_v (int) : The size of the v grid
        """
        r_grid = np.geomspace(2.*m1, right_r*2*m1, n_r)
        v_grid = np.geomspace( np.sqrt(2.* self.potential(r_grid[-1]))/1e3, np.sqrt(2.*self.potential(r_grid[0]))*1e3, n_v)
        R_grid, V_grid = np.meshgrid(r_grid, v_grid)

        # The distribution function depends on the specific energy Eps
        f = lambda v, r: self.f( np.clip(- v**2 /2. + self.potential(r), 0., None))

        # Calculte E_1, F_2, and F_4 for interpolation
        E_1_int = np.zeros(np.shape(V_grid))
        for i, r in enumerate(r_grid):
            for j,v in enumerate(v_grid):
                if v > np.sqrt(2.*self.stellarDistribution.potential(r)):
                    continue
                v_int_grid = np.linspace(v, np.sqrt(2.*self.stellarDistribution.potential(r)), 1000)
                E_1_int[j,i] = simpson(v_int_grid*f(v_int_grid, r), x=v_int_grid)
        E_1_int /=  V_grid
        E_1 = LinearNDInterpolator( list(zip(np.log(R_grid).flatten(), np.log(V_grid).flatten())), # after tests, the logarithmic interpolation seems to work better
                                             E_1_int.flatten())

        F_2_int = np.zeros(np.shape(V_grid))
        for i, r in enumerate(r_grid):
            for j,v in enumerate(v_grid):
                v_int_grid = np.linspace(0, v, 1000)
                F_2_int[j,i] = simpson(v_int_grid**2 *f(v_int_grid, r), x=v_int_grid)
        F_2_int /= V_grid**2
        F_2 = LinearNDInterpolator(list(zip(np.log(R_grid).flatten(), np.log(V_grid).flatten())),
                                            F_2_int.flatten() )

        F_4_int = np.zeros(np.shape(V_grid))
        for i, r in enumerate(r_grid):
            for j,v in enumerate(v_grid):
                v_int_grid = np.linspace(0, v, 1000)
                F_4_int[j,i] = simpson(v_int_grid**4 *f(v_int_grid, r), x=v_int_grid)
        F_4_int /= V_grid**4
        F_4 = LinearNDInterpolator(list(zip(np.log(R_grid).flatten(), np.log(V_grid).flatten())),
                                            F_4_int.flatten() )

        # Make lambda functions for the velocity diffusion coefficients
        E_v_par_pref =  -16.*np.pi**2 * self.CoulombLogarithm # without m2, this is added later
        self.E_v_par = lambda r, v: E_v_par_pref * F_2(np.log(r), np.log(v))

        E_v_par2_pref = 32./3.*np.pi**2 * self.E_m_s**2 * self.CoulombLogarithm
        self.E_v_par2 = lambda r, v: E_v_par2_pref * v * (F_4(np.log(r), np.log(v)) + E_1(np.log(r), np.log(v)))

        E_v_ort2_pref = E_v_par2_pref
        self.E_v_ort2 = lambda r, v: E_v_ort2_pref * v * (3*F_2(np.log(r), np.log(v)) - F_4(np.log(r), np.log(v)) + 2*E_1(np.log(r), np.log(v)))

    def dE_dt(self, hs, ko, opt):
        """
        Calculates the average energy loss due to stellar diffusion.

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
            deps = v *(self.E_m_s**2 + ko.m2 * self.E_m_s)* self.E_v_par(r, v) + self.E_v_par2(r, v) / 2. + self.E_v_ort2(r, v) / 2. # the negative signs cancel out ?
            return deps / (1.+e*np.cos(phi))**2
        dE_dt = 2.* ko.m2 * (1.-e**2)**(3./2.) /2./np.pi * quad(integrand, 0., np.pi, limit=100)[0] # the 1/T is in the 1/2pi
        return - dE_dt

    def dL_dt(self, hs, ko, opt):
        """
        Calculates the average angular momentum loss due to stellar diffusion.

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
            dJ = J / v *(self.E_m_s**2 + ko.m2 * self.E_m_s )* self.E_v_par(r, v) + r**2 / J / 4. * self.E_v_ort2(r, v)
            return dJ / (1.+e*np.cos(phi))**2
        dL_dt = 2.* ko.m2  * (1.-e**2)**(3./2.) /2./np.pi * quad(integrand, 0., np.pi, limit=100)[0]  # the 1/T is in the 1/2pi
        return - dL_dt


    def dEdL_diffusion(self, hs, ko, opt):
        """
        Calculates the matrix for the diffusion term of the SDE due to stellar diffusion.

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
        D_EE = 2.* ko.m2**2 * (1.-e**2)**(3./2.)  /2./np.pi * quad(integrand_deps2, 0., np.pi, limit=60)[0] # The factor of 2 because we only integrate over half the orbit pi

        def integrand_dJ2(phi):
            r, v = ko.get_orbital_parameters(phi)
            dJ2 = J**2 / v**2 * self.E_v_par2(r, v) + 1./2. *( r**2 - J**2/v**2) * self.E_v_ort2(r, v)
            return dJ2 / (1.+e*np.cos(phi))**2
        D_LL = 2.* ko.m2**2 * (1.-e**2)**(3./2.) /2./np.pi * quad(integrand_dJ2, 0., np.pi, limit=60)[0]

        def integrand_dJdeps(phi):
            r, v = ko.get_orbital_parameters(phi)
            dJdeps = J *self.E_v_par2(r, v)
            return dJdeps / (1.+e*np.cos(phi))**2
        D_EL = -2.* ko.m2**2 * (1.-e**2)**(3./2.) /2./np.pi * quad(integrand_dJdeps, 0., np.pi, limit=60)[0]

        sigma = np.zeros((2,2))
        sigma[0,0] =  np.sqrt(D_EE)
        sigma[1,0] =  D_EL / np.sqrt(D_EE)
        sigma[1,1] =  np.sqrt( D_LL - D_EL**2 / D_EE)

        #print(rf"D_EE = {D_EE:.3e}, D_LL = {D_LL:.3e}, D_EL = {D_EL:.3e}, D={np.matmul(sigma, sigma.T)}")
        return sigma

class StellarDiffusionAna(StochasticForce):
    """
    The class for modeling stellar diffusion effects a la https://arxiv.org/pdf/1508.01390.pdf

    This class should give the same result as the above, but here every integral is evaluated on the fly.
    This takes a bit longer but is more accurate than the interpolation

    Attributes:
        stellarDistribution : imripy.halo.MatterHaloDF
            Object describing the stellar distribution
        E_m_s : float
            The first mass moment of the stellar distribution
        CoulombLogarithm : float
            The CoulombLogarithm describing the strength of the scattering
        n : int
            The grid size for the energy grid in integration
    """
    name = "Stellar Diffusion2"

    def __init__(self, hs, stellarDistribution : imripy.halo.MatterHaloDF, E_m_s = c.solar_mass_to_pc,
                 CoulombLogarithm=None):

        super().__init__()
        self.stellarDistribution = stellarDistribution

        self.E_m_s = E_m_s
        self.CoulombLogarithm = CoulombLogarithm or np.log(0.4 * hs.m1 / self.E_m_s)
        self.kappa = (4.*np.pi*self.E_m_s)**2 * self.CoulombLogarithm
        self.n = 200

    def f(self, E):
        return self.stellarDistribution.f(E) / self.E_m_s

    def potential(self, r):
        return self.stellarDistribution.potential(r)

    def v_a(self, r, E_a):
        return np.sqrt(2.*(self.potential(r) - E_a))

    def D_E(self, hs, ko):
        E = ko.m1 / 2. / ko.a
        J = np.sqrt(ko.m1 * ko.a * (1.-ko.e**2))

        def integrand_orbit(phi):
            r, v = ko.get_orbital_parameters(phi)

            E_to_phi = np.linspace(E, self.potential(r), self.n)
            Z_to_E = np.linspace(0, E, self.n)
            int1 = simpson((lambda E_a: self.v_a(r, E_a) / v * self.f(E_a))(E_to_phi), x=E_to_phi)
            int2 = simpson((lambda E_a: self.f(E_a))(Z_to_E), x=Z_to_E)

            deltaE = (ko.m2/self.E_m_s  * int1 - int2)
            return deltaE / (1. + ko.e*np.cos(phi))**2

        return (self.kappa * (1.-ko.e**2)**(3./2.)/2./np.pi
                * quad(integrand_orbit, 0., 2.*np.pi)[0] )

    def dE_dt(self, hs, ko, opt):
        return  -ko.m2 * self.D_E(hs, ko)

    def D_J(self, hs, ko):
        a = ko.a; e = ko.e
        E = ko.m1 / 2. / a
        J = np.sqrt(ko.m1 * a * (1.-e**2))

        def integrand_orbit(phi):
            r, v = ko.get_orbital_parameters(phi)

            E_to_phi = np.linspace(E, self.potential(r), self.n)
            int1 = simpson((lambda E_a: self.v_a(r, E_a) / v * self.f(E_a))(E_to_phi), x=E_to_phi)
            int3 = simpson((lambda E_a: self.v_a(r, E_a)**3 / v**3 * self.f(E_a))(E_to_phi), x=E_to_phi)

            Z_to_E = np.linspace(0, E, self.n)
            int2 = simpson((lambda E_a: self.f(E_a))(Z_to_E), x=Z_to_E)

            deltaJ = ( -J/v**2 * (ko.m2 / self.E_m_s + 1.) * int1
                           + r**2 / 6. / J * ( 2.* int2 + 3*int1 - int3 )
                     )
            return deltaJ / (1. + ko.e*np.cos(phi))**2

        return (self.kappa * (1.-ko.e**2)**(3./2.)/2./np.pi
                    * quad(integrand_orbit, 0., 2.*np.pi)[0]  )

    def dL_dt(self, hs, ko, opt):
        return -ko.m2 * self.D_J(hs, ko)

    def D_EE_(self, hs, ko):
        J = np.sqrt(ko.m1 * ko.a * (1.-ko.e**2))
        E = ko.m1 / 2. / ko.a

        def integrand_deps2(phi):
            r, v = ko.get_orbital_parameters(phi)

            E_to_phi = np.linspace(E, self.potential(r), self.n)
            Z_to_E = np.linspace(0, E, self.n)
            int3 = simpson((lambda E_a: self.v_a(r, E_a)**3 / v**3 * self.f(E_a))(E_to_phi), x=E_to_phi)
            int2 = simpson((lambda E_a: self.f(E_a))(Z_to_E), x=Z_to_E)

            deps2 = 2./3.* v**2 * (int3 + int2)
            return deps2 / (1. + ko.e*np.cos(phi))**2

        return ( self.kappa * (1.-ko.e**2)**(3./2.)/2./np.pi
                 * quad(integrand_deps2, 0., 2.*np.pi)[0] )

    def D_JJ(self, hs, ko):
        J = np.sqrt(ko.m1 * ko.a * (1.-ko.e**2))
        E = ko.m1 / 2. / ko.a

        def integrand_dJ2(phi):
            r, v = ko.get_orbital_parameters(phi)

            E_to_phi = np.linspace(E, self.potential(r), self.n)
            Z_to_E = np.linspace(0, E, self.n)
            int1 = simpson((lambda E_a: self.v_a(r, E_a) / v * self.f(E_a))(E_to_phi), x=E_to_phi)
            int3 = simpson((lambda E_a: self.v_a(r, E_a)**3 / v**3 * self.f(E_a))(E_to_phi), x=E_to_phi)
            int2 = simpson((lambda E_a: self.f(E_a))(Z_to_E), x=Z_to_E)

            dJ2 = 1./v**2 * ( J**2 * int3
                            - J**2 * int1
                            + r**2 * v**2 / 3. * (3.*int1 - int3)
                            + 2./3. * r**2 * v**2 * int2
                     )
            return dJ2 / (1. + ko.e*np.cos(phi))**2

        return (self.kappa * (1.-ko.e**2)**(3./2.) /2./np.pi
                 * quad(integrand_dJ2, 0., 2.*np.pi)[0] )

    def D_EJ(self, hs, ko):
        J = np.sqrt(ko.m1 * ko.a * (1.-ko.e**2))
        E = ko.m1 / 2. / ko.a

        def integrand_dJdeps(phi):
            r, v = ko.get_orbital_parameters(phi)

            E_to_phi = np.linspace(E, self.potential(r), self.n)
            Z_to_E = np.linspace(0, E, self.n)
            int3 = simpson((lambda E_a: self.v_a(r, E_a)**3 / v**3 * self.f(E_a))(E_to_phi), x=E_to_phi)
            int2 = simpson((lambda E_a: self.f(E_a))(Z_to_E), x=Z_to_E)

            dJdeps = -2./3. * J *( int3 + int2)
            return dJdeps / (1. + ko.e*np.cos(phi))**2

        return ( self.kappa * (1.-ko.e**2)**(3./2.)/2./np.pi
                * quad(integrand_dJdeps, 0., 2.*np.pi)[0] )

    def D_EE(self, hs, ko):
        return ko.m2**2 * self.D_EE_(hs, ko)
    def D_LL(self, hs, ko):
        return ko.m2**2 *self.D_JJ(hs, ko)
    def D_EL(self, hs, ko):
        return ko.m2**2 * self.D_EJ(hs, ko)


import numpy as np
from scipy.integrate import quad, quad_vec, odeint
from scipy.interpolate import interp1d, LinearNDInterpolator
import collections

import imripy.constants as c
import imripy.halo


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

    @staticmethod
    def get_relative_velocity(v_m2, v_gas):
        if isinstance(v_m2, tuple):
            if isinstance(v_gas, tuple):
                v_rel = np.sqrt( (v_m2[0] - v_gas[0])**2 + (v_m2[1] - v_gas[1])**2 )
                v_rel *= np.sign(v_m2[1] - v_gas[1]) if np.sign(v_m2[1]) == np.sign(v_gas[1]) else 1.
            else:
                v_rel =  np.sqrt( v_m2[0]**2 + (v_m2[1] - v_gas)**2 )
                v_rel *= np.sign(v_m2[1] - v_gas) if np.sign(v_m2[1]) == np.sign(v_gas) else 1.
        else:
            if isinstance(v_gas, tuple):
                v_rel =  np.sqrt( (v_gas[0])**2 + (v_m2 - v_gas[1])**2 )
                v_rel *= np.sign(v_m2 - v_gas[1]) if np.sign(v_m2) == np.sign(v_gas[1]) else 1.
            else:
                v_rel = v_m2 - v_gas
        return v_rel

    @staticmethod
    def get_orbital_elements(sp, a, e, phi, opt):
        r = a*(1. - e**2)/(1. + e*np.cos(phi))
        v = np.sqrt(sp.m_total(a) *(2./r - 1./a))
        v_phi = r * np.sqrt(sp.m_total(a)*a*(1.-e**2))/r**2
        v_r = np.sqrt(np.max([v**2 - v_phi**2, 0.]))
        # print(r, v, (v_r, v_phi))
        v_phi = v_phi if opt.progradeRotation else -v_phi
        return r, v, v_r, v_phi


    def F(self, sp, r, v, opt):
        """
        Placeholder function that models the dissipative force strength.

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The strength of the dissipative force
        """
        pass


    def dE_dt(self, sp, a, e, opt):
        """
        The function calculates the energy loss due to a force F(r,v) by averaging over a Keplerian orbit
           with semimajor axis a and eccentricity e
        According to https://arxiv.org/abs/1908.10241

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy loss due to accretion
        """
        if  isinstance(a, (collections.Sequence, np.ndarray)):
            return np.array([self.dE_dt(sp, a_i, e, opt) for a_i in a])
        if e == 0.:
            v = sp.omega_s(a)*a
            r, v, v_r, v_phi = self.get_orbital_elements(sp, a, 0., 0., opt)
            return - self.F(sp, a, (v_r, v_phi), opt)*v
        else:
            def integrand(phi):
                r, v, v_r, v_phi = self.get_orbital_elements(sp, a, e, phi, opt)
                return self.F(sp, r, (v_r, v_phi), opt)*v / (1.+e*np.cos(phi))**2
            return -(1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    def dL_dt(self, sp, a, e, opt):
        """
        The function calculates the angular momentum loss due to a force F(r,v) by averaging over a Keplerian orbit
           with semimajor axis a and eccentricity e
        According to https://arxiv.org/abs/1908.10241

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum loss due to accretion
        """
        def integrand(phi):
            r, v, v_r, v_phi = self.get_orbital_elements(sp, a, e, phi, opt)
            return self.F(sp, r, (v_r, v_phi), opt)/v / (1.+e*np.cos(phi))**2
        return -(1.-e**2)**(3./2.)/2./np.pi *np.sqrt(sp.m_total(a) * a*(1.-e**2)) *  quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    def dm2_dt(self, sp, r, v, opt):
        """
        Placeholder function that models the mass gain/loss

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The derivative of the secondary mass
        """
        return 0.


    def dm2_dt_avg(self, sp, a, e, opt):
        """
        The function gives the mass gain due to accretion of the small black hole inside of a halo
           on a Keplerian orbit with semimajor axis a and eccentricity e
        For a circular orbit the dm2_dt function with the corresponding orbital velocity is used
            for an elliptic orbit the average of the expression is used

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The mass gain of the secondary
        """
        dm2_dt = 0.
        if e == 0.:
            v_s = sp.omega_s(a)*a
            dm2_dt = self.dm2_dt(sp, a, v_s, opt)
        else:
            if  isinstance(a, (collections.Sequence, np.ndarray)):
                return np.array([self.dm2_dt_avg(sp, a_i, e, opt) for a_i in a])
            def integrand(phi):
                r = a*(1. - e**2)/(1. + e*np.cos(phi))
                v_s = np.sqrt(sp.m_total(a) *(2./r - 1./a))
                return self.dm2_dt(sp, r, v_s, opt) / (1.+e*np.cos(phi))**2
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


    def dEdL_diffusion(self, sp, a, e, opt):
        """
        Placeholder function that should return the Brownian motion in energy E and angular momentum L.
        The variances of E, L are expected on the diagonal [0,0], [1,1], and the covariance
         on the offdiagonal.

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : np.ndarray with shape=(2,2)
                The diffusion matrix for the SDE
        """
        pass



class GWLoss(DissipativeForce):
    name = "GWLoss"

    def dE_dt(self, sp, a, e, opt):
        """
        The function gives the energy loss due to radiation of gravitational waves
            for a Keplerian orbit with semimajor axis a and eccentricity e
        According to Maggiore (2007)

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy loss due to radiation of gravitational waves of an Keplerian orbit
        """
        return -32./5. * sp.m_reduced(a)**2 * sp.m_total(a)**3 / a**5  / (1. - e**2)**(7./2.) * (1. + 73./24. * e**2 + 37./96. * e**4)

    def dL_dt(self, sp, a, e, opt):
        """
        The function gives the loss of angular momentum due to radiation of gravitational waves of the smaller object
           on a Keplerian orbit with semimajor axis a and eccentricity e
        According to Maggiore (2007)

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum loss due to radiation of gravitational waves
        """
        return -32./5. * sp.m_reduced(a)**2 * sp.m_total(a)**(5./2.) / a**(7./2.)  / (1. - e**2)**2 * (1. + 7./8.*e**2)


class DynamicalFriction(DissipativeForce):
    name = "DynamicalFriction"

    def __init__(self, halo=None, ln_Lambda=-1, relativisticCorrections = False, haloPhaseSpaceDescription = False, includeHigherVelocities = True):
        self.halo = halo
        self.ln_Lambda = ln_Lambda
        self.relativisticCorrections = relativisticCorrections
        self.haloPhaseSpaceDescription = haloPhaseSpaceDescription
        self.dmPhaseSpaceFraction = dmPhaseSpaceFraction
        self.v_max = v_max

    def F(self, sp, r, v, opt):
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
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float or tuple)   : The speed of the orbiting object, either as |v|, or (v_r, v_theta) split into the direction of r, theta
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the dynamical friction force
        """
        ln_Lambda = self.ln_Lambda
        halo = self.halo or sp.halo
        v_gas = halo.velocity(r)
        v_rel = ( self.get_relative_velocity(v, v_gas) if opt.considerRelativeVelocities
                        else self.get_relative_velocity(v, 0.) )
        # print(v, v_gas, v_rel)

        if ln_Lambda < 0.:
            ln_Lambda = np.log(sp.m1/sp.m2)/2.

        relCovFactor = 1.
        if self.relativisticCorrections:
            relCovFactor = (1. + v_rel**2)**2 / (1. - v_rel**2)

        if self.haloPhaseSpaceDescription:
            if 'v_max' in opt.additionalParameters:
                v_max = opt.additionalParameters['v_max']
            else:
                v_max = self.v_max
            density = halo.density(r, v_max=(v_max if not v_max is None else np.abs(v_rel)))
        else:
            density = halo.density(r) * self.dmPhaseSpaceFraction

        F_df = 4.*np.pi * relCovFactor * sp.m2**2 * density * ln_Lambda / v_rel**2  * np.sign(v_rel)
        return np.nan_to_num(F_df)


class GasDynamicalFriction(DissipativeForce):
    name = "GasDynamicalFriction"

    def __init__(self, disk = None, ln_Lambda= -1., frictionModel='Ostriker'):
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

    def F(self, sp, r, v, opt):
        """
        The function gives the force of the dynamical friction of an object inside a gaseous disk at radius r
            and with velocity v

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float or tuple)   : The speed of the orbiting object, either as |v|, or (v_r, v_theta) split into the direction of r, theta
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the dynamical friction force
        """
        ln_Lambda = self.ln_Lambda
        disk = self.disk or sp.baryonicHalo
        v_gas = disk.velocity(r)
        v_rel = ( self.get_relative_velocity(v, v_gas) if opt.considerRelativeVelocities
                        else self.get_relative_velocity(v, 0.) )
        # print(v, v_gas, v_rel)

        if ln_Lambda < 0.:
            ln_Lambda = np.log(sp.m1/sp.m2)/2.

        if self.frictionModel == 'Ostriker':
            c_s = disk.soundspeed(r)
            I = np.where( np.abs(v_rel) >= c_s,
                                1./2. * np.log(1. - (c_s/np.abs(v_rel))**2) + ln_Lambda, # supersonic regime
                                1./2. * np.log((1. + np.abs(v_rel)/c_s)/(1. - np.abs(v_rel)/c_s)) - np.abs(v_rel)/c_s) # subsonic regime
            ln_Lambda = I
        elif self.frictionModel == 'Sanchez-Salcedo':
                H = disk.scale_height(r)
                R_acc = 2.*sp.m2 /v_rel**2
                ln_Lambda =  7.15*H/R_acc

        F_df = 4.*np.pi * sp.m2**2 * disk.density(r) * ln_Lambda / v_rel**2  * np.sign(v_rel)
        #print(v, v_gas, v_rel, F_df)
        return np.nan_to_num(F_df)


class AccretionLoss(DissipativeForce):
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

    def BH_cross_section(self, sp, r, v_rel, opt):
        """
        The function gives the cross section of a small black hole (m2) moving through a halo of particles
        Choose model through opt.accretionModel parameter
            'Collisionless' (default): according to https://arxiv.org/pdf/1711.09706.pdf
            'Bondi-Hoyle' : according to https://arxiv.org/pdf/1302.2646.pdf

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system, the small black hole is taken to be sp.m2
            v_rel  (float)      : The relative velocity of the black hole to the halo
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The black hole cross section
        """
        halo = self.halo or sp.halo
        if self.accretionModel == 'Bondi-Hoyle':

            dm_soundspeed2 = halo.soundspeed(r)**2 if self.withSoundspeed else 0.

            return 4.*np.pi * sp.m2**2 / (v_rel**2 +  dm_soundspeed2)**(3./2.)  / v_rel

        elif self.accretionModel == 'Collisionless':
            return (np.pi * sp.m2**2. / v_rel**2.) * (8. * (1. - v_rel**2.))**3 / (4. * (1. - 4. * v_rel**2. + (1. + 8. * v_rel**2.)**(1./2.)) * (3. - (1. + 8. * v_rel**2.)**(1./2.))**2.)
            #return 16. * np.pi * sp.m2**2 / v**2  * (1. + v**2)


    def dm2_dt(self, sp, r, v, opt):
        """
        The function gives the mass gain due to accretion of the small black hole inside of the dark matter halo
           for a small black hole with relative velocity v to the halo at radius r
        The equation of https://arxiv.org/pdf/1711.09706.pdf is used

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radial position of the black hole in the halo
            v  (float or tuple)   : The speed of the orbiting object, either as |v|, or (v_r, v_theta) split into the direction of r, theta
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The mass gain due to accretion
        """
        halo = self.halo or sp.halo
        v_gas = halo.velocity(r)
        v_rel = ( self.get_relative_velocity(v, v_gas) if opt.considerRelativeVelocities
                        else self.get_relative_velocity(v, 0.) )
        v = np.sqrt( v[0]**2 + v[1]**2 ) if isinstance(v, tuple) else v
        return halo.density(r) * v * self.BH_cross_section(sp, r, np.abs(v_rel), opt)


    def F(self, sp, r, v, opt):
        """
        The function gives the total force acting on an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v through accretion effects

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float or tuple)   : The speed of the orbiting object, either as |v|, or (v_r, v_theta) split into the direction of r, theta
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the accretion force
        """
        F_acc = self.F_acc(sp, r, v, opt)
        F_acc_rec = self.F_acc_recoil(sp, r, v, opt) if self.includeRecoil else 0.
        return F_acc + F_acc_rec

    def F_acc(self, sp, r, v, opt):
        """
        The function gives the force mimicing the accretion effects of an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float or tuple)   : The speed of the orbiting object, either as |v|, or (v_r, v_theta) split into the direction of r, theta
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the accretion force
        """
        halo = self.halo or sp.halo
        v_gas = halo.velocity(r)
        v_rel = ( self.get_relative_velocity(v, v_gas) if opt.considerRelativeVelocities
                        else self.get_relative_velocity(v, 0.) )
        v = np.sqrt( v[0]**2 + v[1]**2 ) if isinstance(v, tuple) else v
        return self.dm2_dt(sp, r, np.abs(v_rel), opt) *  v


    def F_recoil(self, sp, r, v, opt):
        """
        The function gives the recoil force of the accretion of an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float or tuple)   : The speed of the orbiting object, either as |v|, or (v_r, v_theta) split into the direction of r, theta
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the accretion recoil force
        """
        halo = self.halo or sp.halo
        v_gas = halo.velocity(r)
        v_rel = ( self.get_relative_velocity(v, v_gas) if opt.considerRelativeVelocities
                        else self.get_relative_velocity(v, 0.) )
        v = np.sqrt( v[0]**2 + v[1]**2 ) if isinstance(v, tuple) else v
        return self.dm2_dt(sp, r, np.abs(v_rel), opt) * v


class GasInteraction(DissipativeForce):
    name = "GasInteraction"

    def __init__(self, disk = None, gasInteraction = 'gasTorqueLossTypeI', alpha=0.1, fudgeFactor=1.):
        if not gasInteraction in ['gasTorqueLossTypeI', 'gasTorqueLossTypeII']:
            raise Exception(f"Gas Interaction type not recognized: {gasInteraction}")
        self.gasInteraction = gasInteraction
        self.alpha = None
        self.fudgeFactor = fudgeFactor
        self.disk = disk

    def F(self, sp, r, v, opt):
        """
        The function gives the force an accretion disk would exert on a smaller black hole for different models of gas interaction

        Choose model through opt.gasInteraction parameter
            'gasTorqueLoss' : according to https://arxiv.org/pdf/2206.05292.pdf
            'dynamicalFriction' : according to https://arxiv.org/pdf/2006.10206.pdf

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float or tuple)   : The speed of the orbiting object, either as |v|, or (v_r, v_theta) split into the direction of r, theta
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the force through gas interactions
        """
        disk = self.disk or sp.baryonicHalo
        v_gas = disk.velocity(r)
        v_rel = ( self.get_relative_velocity(v, v_gas) if opt.considerRelativeVelocities
                        else self.get_relative_velocity(v, 0.) )

        if self.gasInteraction == 'gasTorqueLossTypeI':
            mach_number = disk.mach_number(r)
            Sigma = disk.surface_density(r)
            Omega = sp.omega_s(r)
            Gamma_lin = Sigma*r**4 * Omega**2 * (sp.m2/sp.m1)**2 * mach_number**2

            F_gas = Gamma_lin * sp.m2/sp.m1 / r

        elif self.gasInteraction == 'gasTorqueLossTypeII':
            mach_number = disk.mach_number(r)
            Sigma = disk.surface_density(r)
            alpha = disk.alpha if hasattr(disk, 'alpha') else self.alpha # requires ShakuraSunyaevDisk atm

            Omega = sp.omega_s(r)
            Gamma_vis = 3.*np.pi * alpha * Sigma * r**4 * Omega**2 / mach_number**2 if mach_number > 0. else 0.

            fudge_factor = self.fudgeFactor
            Gamma_gas = fudge_factor * Gamma_vis

            F_gas = Gamma_gas * sp.m2/sp.m1 / r

        '''
        elif self.gasInteraction == 'gasTorqueLossTypeITanaka':
            C = 7.2e-10
            n_r = 8.
            n_alpha = -1.
            n_fedd = -3.
            n_M1 = 1.
            A = (C * (sp.halo.alpha/0.1)**n_alpha * (sp.halo.f_edd/sp.halo.eps)**n_fedd
                    * (sp.m1/1e6/c.solar_mass_to_pc)**n_M1 )
            L0 = 32./5. * sp.m2 / sp.m1 * (r/sp.m1)**(-7./2.)
            L_disk = A * (r/10./sp.m1)**n_r * L0

            F_gas = L_disk * sp.m2/sp.m1 / r

        elif self.gasInteraction == 'hydrodynamicDragForce':
            C_d = 1.
            R_p = 2.*sp.m2
            rho = sp.halo.density(r)

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

    def __init__(self, stellarDistribution : imripy.halo.MatterHaloDF, sp, CoulombLogarithm=None, E_m_s = c.solar_mass_to_pc, E_m_s2 = c.solar_mass_to_pc**2, m=None):
        """
        The constructor for the StellarDiffusion class
        The values are initialized according to https://arxiv.org/pdf/2304.13062.pdf if not provided

        Parameters:
            istellarDistribution : imripy.halo.MatterHaloDF
                Object describing the stellar distribution
            sp : imripy.merger_system.SystemProp
                The system prop object describing the merger system
            E_m_s : (optional) float
                The first mass moment of the stellar distribution
            E_m_s2 :(optional)  float
                The second mass moment of the stellar distribution
            CoulombLogarithm : (optional) float
                The CoulombLogarithm describing the strength of the scattering - either this or sp must be provided
            m : (optional) float
                The mass of the secondary object subject to stellar diffusion - either this or sp must be provided
        """
        super().__init__()
        self.stellarDistribution = stellarDistribution

        self.E_m_s = E_m_s
        self.E_m_s2 = E_m_s2
        self.m = m or sp.m2
        self.CoulombLogarithm = CoulombLogarithm or np.log(0.4 * sp.m1 / self.E_m_s)

        self.calc_velocity_diffusion_coeff(sp)



    def calc_velocity_diffusion_coeff(self, sp):
        """
        Calculates the velocity diffusion coefficients and saves them in the class for later use as a lambda function.
        This should only be needed once (or when the distribution function stellarDistribution.f changes)
        Eq (24)-(28) from https://arxiv.org/pdf/2304.13062.pdf
        """

        r_grid = np.geomspace(2.*sp.m1, 1e8*2*sp.m1, 60)
        v_grid = np.geomspace( np.sqrt(2.* self.stellarDistribution.potential(r_grid[-1])/10.), np.sqrt(2.*self.stellarDistribution.potential(r_grid[0])), 61)
        R_grid, V_grid = np.meshgrid(r_grid, v_grid)

        # The distribution function depends on the specific energy Eps
        f = lambda v, r: self.stellarDistribution.f( np.max([- v**2 /2. + self.stellarDistribution.potential(r), 0.]))

        # Calculte E_1, F_2, and F_4 for interpolation
        E_1_int = np.zeros(np.shape(V_grid))
        for i, r in enumerate(r_grid):
            E_1_int[:,i] = quad( lambda v_int : v_int * f(v_int, r), 0, np.inf, limit=100)[0] * np.ones(np.shape(v_grid))
        E_1 = LinearNDInterpolator( list(zip(R_grid.flatten(), V_grid.flatten())),
                                             (E_1_int / V_grid).flatten(), rescale=True)

        F_2_int = np.zeros(np.shape(V_grid))
        for i, r in enumerate(r_grid):
            F_2_int[:,i] = odeint( lambda a, v_int : v_int**2 * f(v_int, r), 0, v_grid )[:,0]
        F_2 = LinearNDInterpolator(list(zip(R_grid.flatten(), V_grid.flatten())),
                                    (F_2_int / V_grid**2).flatten() , rescale = True)

        F_4_int = np.zeros(np.shape(V_grid))
        for i, r in enumerate(r_grid):
            F_4_int[:,i] = odeint( lambda a, v_int : v_int**4 * f(v_int, r), 0, v_grid )[:,0]
        F_4 = LinearNDInterpolator(list(zip(R_grid.flatten(), V_grid.flatten())),
                                   (F_4_int / V_grid**4).flatten() , rescale = True)

        print(R_grid, V_grid, E_1_int, F_2_int, F_4_int)

        # Make lambda functions for the velocity diffusion coefficients
        E_v_par_pref =  -16.*np.pi**2 *(self.E_m_s2 + self.m * self.E_m_s ) * self.CoulombLogarithm
        self.E_v_par = lambda r, v: E_v_par_pref * F_2(r, v)

        E_v_par2_pref = 32./3.*np.pi**2 * self.E_m_s2 * self.CoulombLogarithm
        self.E_v_par2 = lambda r, v: E_v_par2_pref * v * (F_4(r, v) + E_1(r, v))

        E_v_ort2_pref = E_v_par2_pref
        self.E_v_ort2 = lambda r, v: E_v_ort2_pref * v * (3*F_2(r, v) - F_4(r, v) + 2*E_1(r, v))



    def dE_dt(self, sp, a, e, opt):
        """
        Calculates the average energy loss due to stellar diffusion.
        Eq (14) from https://arxiv.org/pdf/2304.13062.pdf

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy loss due to stellar diffusion
        """
        def integrand(phi):
            r, v, v_r, v_phi = self.get_orbital_elements(sp, a, e, phi, opt)
            deps = v * self.E_v_par(r, v) + self.E_v_par2(r, v) / 2. + self.E_v_ort2(r, v) / 2.
            #print(v, deps, self.E_v_par(v), self.E_v_par2(v), self.E_v_ort2(v) )
            return deps / (1.+e*np.cos(phi))**2
        #plt.plot(np.linspace(0, np.pi), np.array([integrand(phi) for phi in np.linspace(0, np.pi)]))
        dE_dt = (1.-e**2)**(3./2.) * quad(integrand, 0., np.pi, limit=50)[0]
        return - 2.* self.m * dE_dt

    def dL_dt(self, sp, a, e, opt):
        """
        Calculates the average angular momentum loss due to stellar diffusion.
        Eq (16) from https://arxiv.org/pdf/2304.13062.pdf

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum loss due to stellar diffusion
        """
        J = np.sqrt(sp.m1**2 / sp.m_total() * a * (1.-e**2))
        def integrand(phi):
            r, v, v_r, v_phi = self.get_orbital_elements(sp, a, e, phi, opt)
            dJ = J / v * self.E_v_par(r, v) + r**2 / J / 4. * self.E_v_ort2(r, v)
            return dJ / (1.+e*np.cos(phi))**2
        dL_dt = (1.-e**2)**(3./2.) * quad(integrand, 0., np.pi, limit=50)[0]
        return -2.* self.m * dL_dt

    def dEdL_diffusion(self, sp, a, e, opt):
        """
        Calculates the matrix for the diffusion term of the SDE due to stellar diffusion.
        The variances are on the diagonal according to Eq(15) and (17), the covariance is on the off-diagonal Eq(18)

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : np.matrix
                The diffusion matrix
        """
        J = np.sqrt(sp.m1**2 / sp.m_total() * a * (1.-e**2))

        covmatrix = np.zeros((2,2))
        def integrand_deps2(phi):
            r, v, v_r, v_phi = self.get_orbital_elements(sp, a, e, phi, opt)
            deps2 = v**2 *  self.E_v_par2(r, v)
            return deps2 / (1.+e*np.cos(phi))**2
        covmatrix[0,0] = quad(integrand_deps2, 0., np.pi, limit=60)[0]

        def integrand_dJ2(phi):
            r, v, v_r, v_phi = self.get_orbital_elements(sp, a, e, phi, opt)
            dJ2 = J**2 / v**2 * self.E_v_par2(r, v) + 1./2. *( r**2 - J**2/v**2) * self.E_v_ort2(r, v)
            return dJ2 / (1.+e*np.cos(phi))**2
        covmatrix[1,1] = quad(integrand_dJ2, 0., np.pi, limit=60)[0]

        def integrand_dJdeps(phi):
            r, v, v_r, v_phi = self.get_orbital_elements(sp, a, e, phi, opt)
            dJdeps = J *self.E_v_par2(r, v)
            return dJdeps / (1.+e*np.cos(phi))**2
        covmatrix[0,1] = quad(integrand_dJdeps, 0., np.pi, limit=60)[0]
        covmatrix[1,0] = covmatrix[0,1]

        return -2.* self.m * (1.-e**2)**(3./2.) * covmatrix



import numpy as np
from scipy.integrate import solve_ivp, quad, simpson
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from scipy.special import ellipeinc, ellipe, ellipkinc
from scipy.spatial import Delaunay
import collections.abc
#import sys
import time
import imripy.constants as c
import imripy.merger_system as ms
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

    def get_relative_velocity(v_m2, v_gas):
        if isinstance(v_m2, tuple):
            v_rel = (np.sign(v_m2[1]-v_gas[1])* np.sqrt( (v_m2[0] - v_gas[0])**2 + (v_m2[1] - v_gas[1])**2 ) if isinstance(v_gas, tuple)
                                                        else np.sign(v_m2[1]-v_gas) * np.sqrt( v_m2[0]**2 + (v_m2[1] - v_gas)**2 ) )
        else:
            v_rel = (np.sign(v_m2-v_gas[1]) * np.sqrt( (v_gas[0])**2 + (v_m2 - v_gas[1])**2 ) if isinstance(v_gas, tuple)
                                                        else  v_m2 - v_gas )
        return v_rel


    def get_orbital_elements(sp, a, e, phi):
        r = a*(1. - e**2)/(1. + e*np.cos(phi))
        v = np.sqrt(sp.m_total(a) *(2./r - 1./a))
        v_phi = r * np.sqrt(sp.m_total(a)*a*(1.-e**2))/r**2
        v_r = np.sqrt(np.max([v**2 - v_phi**2, 0.]))
        # print(r, v, (v_r, v_phi))
        return r, v, v_r, v_phi


    @classmethod
    def F(cls, sp, r, v, opt):
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
        return 0.


    @classmethod
    def dE_dt(cls, sp, a, e, opt):
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
        if e == 0.:
            v = sp.omega_s(a)*a
            return - cls.F(sp, a, v, opt)*v
        else:
            if  isinstance(a, (collections.Sequence, np.ndarray)):
                return np.array([cls.dE_dt(sp, a_i, e, opt) for a_i in a])
            def integrand(phi):
                r, v, v_r, v_phi = cls.get_orbital_elements(sp, a, e, phi)
                return cls.F(sp, r, (v_r, v_phi), opt)*v / (1.+e*np.cos(phi))**2
            return -(1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    @classmethod
    def dL_dt(cls, sp, a, e, opt):
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
            r, v, v_r, v_phi = cls.get_orbital_elements(sp, a, e, phi)
            return cls.F(sp, r, (v_r, v_phi), opt)/v / (1.+e*np.cos(phi))**2
        return -(1.-e**2)**(3./2.)/2./np.pi *np.sqrt(sp.m_total(a) * a*(1.-e**2)) *  quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    @classmethod
    def dm2_dt_avg(cls, sp, a, e, opt):
        """
        Placeholder function that models the mass gain of the secondary

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The mass gain of the secondary
        """
        return 0.


class GWLoss(DissipativeForce):

    @classmethod
    def dE_dt(cls, sp, a, e, opt):
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

    @classmethod
    def dL_dt(cls, sp, a, e, opt):
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

    @classmethod
    def F(cls, sp, r, v, opt):
        """
        The function gives the force of the dynamical friction of an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v
        The opt.ln_Lambda is the Coulomb logarithm, for which different authors use different values. Set to -1 so that Lambda = sqrt(m1/m2)
        The opt.relativisticDynamicalFrictionCorrections parameter allows the use of the correction factor as given by eq (15) of
                https://arxiv.org/pdf/2204.12508.pdf ( except for the typo in the gamma factor )
        The opt.useHaloPhaseSpaceDescription parameter allows to use not the total dark matter density at r, but uses the halo phase space description
            such that only particles below a given v_max scatter. This option requires sp.halo to be of type DynamicSS.
            v_max can be provided via opt.additionalParameters['v_max']. If v_max is None, it is taken to be the orbital velocity.
        The 'useGasDynamicalFrictionDescription' parameter allows alternative descriptions of dynamical friction inside gaseous mediums
            'Ostriker' refers to eq (3) of https://arxiv.org/pdf/2203.01334.pdf
            'Sanchez-Salcedo' refers to eq (5) of https://arxiv.org/pdf/2006.10206.pdf

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float or tuple)   : The speed of the orbiting object, either as |v|, or (v_r, v_theta) split into the direction of r, theta
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the dynamical friction force
        """
        ln_Lambda = opt.ln_Lambda
        v_gas = sp.halo.velocity(r)
        v_rel = ( cls.get_relative_velocity(v, v_gas) if ('considerRelativeVelocities' in opt.additionalParameters and opt.additionalParameters['considerRelativeVelocities'])
                        else cls.get_relative_velocity(v, 0.) )
        # print(v, v_gas, v_rel)

        if ln_Lambda < 0.:
            ln_Lambda = np.log(sp.m1/sp.m2)/2.

        if 'useGasDynamicalFrictionDescription' in opt.additionalParameters:
            if opt.additionalParameters['useGasDynamicalFrictionDescription'] == 'Ostriker':
                c_s = sp.halo.soundspeed(r)
                I = np.where( np.abs(v_rel) >= c_s,
                                    1./2. * np.log(1. - (c_s/np.abs(v_rel))**2) + ln_Lambda, # supersonic regime
                                    1./2. * np.log((1. + np.abs(v_rel)/c_s)/(1. - np.abs(v_rel)/c_s)) - np.abs(v_rel)/c_s) # subsonic regime
                ln_Lambda = I
                #ln_Lambda = 0.5
                #print(c_s, v_rel, I, ln_Lambda)
            elif opt.additionalParameters['useGasDynamicalFrictionDescription'] == 'Sanchez-Salcedo':
                H = sp.halo.scale_height(r)
                R_acc = 2.*sp.m2 /v_rel**2
                ln_Lambda =  7.15*H/R_acc


        relCovFactor = 1.
        if 'relativisticDynamicalFrictionCorrections' in opt.additionalParameters and opt.additionalParameters['relativisticDynamicalFrictionCorrections']:
            relCovFactor = (1. + v_rel**2)**2 / (1. - v_rel**2)

        if opt.haloPhaseSpaceDescription:
            density = sp.halo.density(r, v_max=(opt.additionalParameters['v_max'] if 'v_max' in opt.additionalParameters else np.abs(v_rel)))
        else:
            density = sp.halo.density(r) * opt.dmPhaseSpaceFraction

        F_df = 4.*np.pi * relCovFactor * sp.m2**2 * density * ln_Lambda / v_rel**2  * np.sign(v_rel)
        return np.nan_to_num(F_df)


class AccretionLoss(DissipativeForce):

    def BH_cross_section(sp, r, v, opt):
        """
        The function gives the cross section of a small black hole (m2) moving through a halo of particles
        Choose model through opt.accretionModel parameter
            'Collisionless' (default): according to https://arxiv.org/pdf/1711.09706.pdf
            'Bondi-Hoyle' : according to https://arxiv.org/pdf/1302.2646.pdf

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system, the small black hole is taken to be sp.m2
            v  (float)      : The relative velocity of the black hole to the halo
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The black hole cross section
        """
        if opt.accretionModel == 'Bondi-Hoyle':
            if hasattr(sp.halo, 'soundspeed'):
                dm_soundspeed2 = sp.halo.soundspeed(r)**2
            elif 'dm_soundspeed2' in opt.additionalParameters:
                dm_soundspeed2 = opt.additionalParameters['dm_soundspeed2']
            else:
                dm_soundspeed2 = 0.
            v_halo_r, v_halo_phi = sp.halo.velocity(r) if hasattr(sp.halo, 'velocity') else [0.,0.]
            delta_v = np.sqrt(v**2 + v_halo_r**2)  # TODO: Improve!!
            #print(v, v_halo_r, delta_v)
            return 4.*np.pi * sp.m2**2 / (delta_v**2 +  dm_soundspeed2)**(3./2.)  / delta_v

        return (np.pi * sp.m2**2. / v**2.) * (8. * (1. - v**2.))**3 / (4. * (1. - 4. * v**2. + (1. + 8. * v**2.)**(1./2.)) * (3. - (1. + 8. * v**2.)**(1./2.))**2.)
        #return 16. * np.pi * sp.m2**2 / v**2  * (1. + v**2)

    @classmethod
    def dm2_dt(cls, sp, r, v, opt):
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
        v_gas = sp.halo.velocity(r)
        v_rel = ( cls.get_relative_velocity(v, v_gas) if ('considerRelativeVelocities' in opt.additionalParameters and opt.additionalParameters['considerRelativeVelocities'])
                        else cls.get_relative_velocity(v, 0.) )
        v = np.sqrt( v[0]**2 + v[1]**2 ) if isinstance(v, tuple) else v
        return sp.halo.density(r) * v * cls.BH_cross_section(sp, r, np.abs(v_rel), opt)


    @classmethod
    def dm2_dt_avg(cls, sp, a, e, opt):
        """
        The function gives the mass gain due to accretion of the small black hole inside of the dark matter halo
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
                The mass gain due to accretion on an orbit
        """
        dm2_dt = 0.
        if opt.accretion:
            if e == 0.:
                v_s = sp.omega_s(a)*a
                dm2_dt = cls.dm2_dt(sp, a, v_s, opt)
            else:
                if  isinstance(a, (collections.Sequence, np.ndarray)):
                    return np.array([cls.dm2_dt_avg(sp, a_i, e, opt) for a_i in a])
                def integrand(phi):
                    r = a*(1. - e**2)/(1. + e*np.cos(phi))
                    v_s = np.sqrt(sp.m_total(a) *(2./r - 1./a))
                    return cls.dm2_dt(sp, r, v_s, opt) / (1.+e*np.cos(phi))**2
                dm2_dt = (1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]

        dm2_baryons_dt = 0.
        if opt.baryonicHaloEffects and opt.baryonicEvolutionOptions.accretion:
            dmHalo = sp.halo
            sp.halo = sp.baryonicHalo
            dm2_baryons_dt = cls.dm2_dt_avg(sp, a, e, opt.baryonicEvolutionOptions)
            sp.halo = dmHalo

        if(opt.verbose > 2):
            print(f"dm2_dt={dm2_dt}, dm2_baryons_dt = {dm2_baryons_dt}")
        return dm2_dt + dm2_baryons_dt


    @classmethod
    def F(cls, sp, r, v, opt):
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
        v_gas = sp.halo.velocity(r)
        v_rel = ( cls.get_relative_velocity(v, v_gas) if ('considerRelativeVelocities' in opt.additionalParameters and opt.additionalParameters['considerRelativeVelocities'])
                        else cls.get_relative_velocity(v, 0.) )
        v = np.sqrt( v[0]**2 + v[1]**2 ) if isinstance(v, tuple) else v
        return cls.dm2_dt(sp, r, np.abs(v_rel), opt) *  v


    @classmethod
    def F_recoil(cls, sp, r, v, opt):
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
        v_gas = sp.halo.velocity(r)
        v_rel = ( cls.get_relative_velocity(v, v_gas) if ('considerRelativeVelocities' in opt.additionalParameters and opt.additionalParameters['considerRelativeVelocities'])
                        else cls.get_relative_velocity(v, 0.) )
        v = np.sqrt( v[0]**2 + v[1]**2 ) if isinstance(v, tuple) else v
        return cls.dm2_dt(sp, r, np.abs(v_rel), opt) * v


def GasInteraction(DissipativeForce):

    @classmethod
    def F(cls, sp, r, v, opt):
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
        v_gas = sp.halo.velocity(r)
        v_rel = ( cls.get_relative_velocity(v, v_gas) if ('considerRelativeVelocities' in opt.additionalParameters and opt.additionalParameters['considerRelativeVelocities'])
                        else cls.get_relative_velocity(v, 0.) )

        if opt.additionalParameters['gasInteraction'] == 'gasTorqueLossTypeI':
            mach_number = sp.halo.mach_number(r)
            Sigma = sp.halo.surface_density(r)
            Omega = sp.omega_s(r)
            Gamma_lin = Sigma*r**4 * Omega**2 * (sp.m2/sp.m1)**2 * mach_number**2

            F_gas = Gamma_lin * sp.m2/sp.m1 / r

        elif opt.additionalParameters['gasInteraction'] == 'gasTorqueLossTypeII':
            mach_number = sp.halo.mach_number(r)
            Sigma = sp.halo.surface_density(r)
            if hasattr(sp.halo, 'alpha'):
                alpha = sp.halo.alpha  # requires ShakuraSunyaevDisk atm
            else:
                alpha = opt.additionalParameters['gasTorqueAlpha'] if 'gasTorqueAlpha' in opt.additionalParameters else 0.1

            Omega = sp.omega_s(r)
            Gamma_vis = 3.*np.pi * alpha * Sigma * r**4 * Omega**2 / mach_number**2 if mach_number > 0. else 0.

            fudge_factor = opt.additionalParameters['gasTorqueFudgeFactor'] if 'gasTorqueFudgeFactor' in opt.additionalParameters else 1.
            Gamma_gas = fudge_factor * Gamma_vis

            F_gas = Gamma_gas * sp.m2/sp.m1 / r

        '''
        elif opt.additionalParameters['gasInteraction'] == 'gasTorqueLossTypeITanaka':
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

        elif opt.additionalParameters['gasInteraction'] == 'hydrodynamicDragForce':
            C_d = 1.
            R_p = 2.*sp.m2
            rho = sp.halo.density(r)

            F_gas = 1./2. * C_d * np.pi * rho * v_rel**2 * R_p**2
        '''

        return F_gas


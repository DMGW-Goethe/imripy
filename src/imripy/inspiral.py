import numpy as np
from scipy.integrate import solve_ivp, quad, simpson
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from scipy.special import ellipeinc, ellipe, ellipkinc
from scipy.spatial import Delaunay
import collections.abc
#import sys
import time
import imripy.merger_system as ms
import imripy.halo



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
            gwEmissionLoss : bool
                Whether to include energy losses by graviational waves
            dynamicalFrictionLoss : bool
                Whether to include energy losses by dynamical friction
            accretion : bool
                Whether to include accretion effects and evolve the secondary mass
            accretionForceLoss : bool
                Whether to include the energy loss due to the accretion mass change
            accretionRecoilLoss : bool
                Whether to include the energy loss due to the accretion recoil
            baryonicHaloEffects : bool
                Whether to include the effects of a baryonic halo. This requires sp.baryonicHalo to be not None
            baryonicEvolutionOptions : EvolutionOptions
                The evolution Options to describe the interaction with the baryon halo. Be careful to avoid nesting!
            haloPhaseSpaceDescription : bool
                Whether to use the phase space description of the halo to calculate relative velocities
                This requires the SystemProp.halo to be of type DynamicSS

        """
        def __init__(self, accuracy=1e-8, verbose=1, elliptic=True, gwEmissionLoss=True, dynamicalFrictionLoss=True, accretion=False,
                                    accretionForceLoss=True, accretionRecoilLoss=True, accretionModel='',
                                    baryonicHaloEffects=False, baryonicEvolutionOptions=None,
                                    haloPhaseSpaceDescription=False, dmPhaseSpaceFraction=1., coulombLog=-1.,
                                    **kwargs):
            self.accuracy = accuracy
            self.verbose = verbose
            self.elliptic = elliptic
            self.gwEmissionLoss = gwEmissionLoss
            self.dynamicalFrictionLoss = dynamicalFrictionLoss
            self.accretion = accretion
            self.accretionForceLoss = accretionForceLoss and accretion
            self.accretionRecoilLoss = accretionRecoilLoss and accretion
            self.accretionModel = accretionModel if accretionModel in ['Classic', 'Bondi-Hoyle'] else 'Classic'
            self.baryonicHaloEffects = baryonicHaloEffects
            self.baryonicEvolutionOptions = baryonicEvolutionOptions
            self.haloPhaseSpaceDescription = haloPhaseSpaceDescription
            self.additionalParameters = kwargs
            self.ln_Lambda = coulombLog
            self.dmPhaseSpaceFraction = dmPhaseSpaceFraction

            if not self.baryonicEvolutionOptions is None:
                self.baryonicEvolutionOptions.baryonicHaloEffects = False
                self.baryonicEvolutionOptions.baryonicEvolutionOptions = None
                self.baryonicEvolutionOptions.gwEmissionLoss = False


        def __str__(self):
            s = "Options: "
            if not self.gwEmissionLoss:
                s += f"gwEmissionLoss = {self.gwEmissionLoss},"
            if not self.dynamicalFrictionLoss:
                s += f" dynamicalFrictionLoss = {self.dynamicalFrictionLoss},"
            s += f"accretion = {self.accretion}"
            if self.accretion:
                s += f" (accretionForceLoss = {self.accretionForceLoss}, accretionRecoilLoss = {self.accretionRecoilLoss}, accretionModel = {self.accretionModel})"
            s += f", haloPhaseSpaceDescription = {self.haloPhaseSpaceDescription}"
            s += f", accuracy = {self.accuracy:.1e}"
            if self.baryonicHaloEffects:
                s += f", baryonicHaloEffects = {self.baryonicHaloEffects}"
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
        return np.sqrt(a * (1-e**2) * sp.m_total(a) * sp.m_reduced(a)**2 )
        #return np.sqrt( -(1. - e**2) * sp.m_reduced(a)**3 * sp.m_total(a)**2 / 2. / Classic.E_orbit(sp, a, e))


    def dE_gw_dt(sp, a, e=0., opt=EvolutionOptions()):
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


    def dL_gw_dt(sp, a, e, opt=EvolutionOptions()):
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


    def F_df(sp, r, v, opt=EvolutionOptions()):
        """
        The function gives the force of the dynamical friction of an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v
        The opt.ln_Lambda is the Coulomb logarithm, for which different authors use different values. Set to -1 so that Lambda = sqrt(m1/m2)
        The opt.relativisticDynamicalFrictionCorrections parameter allows the use of the correction factor as given by eq (15) of
                https://arxiv.org/pdf/2204.12508.pdf ( except for the typo in the gamma factor )
        The opt.useHaloPhaseSpaceDescription parameter allows to use not the total dark matter density at r, but uses the halo phase space description
            such that only particles below a given v_max scatter. This option requires sp.halo to be of type DynamicSS.
            v_max can be provided via opt.additionalParameters['v_max']. If v_max is None, it is taken to be the orbital velocity.

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float)      : The speed of the orbiting object wrt to the dark matter halo
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the dynamical friction force
        """
        ln_Lambda = opt.ln_Lambda
        if ln_Lambda < 0.:
            ln_Lambda = np.log(sp.m1/sp.m2)/2.
        relCovFactor = 1.
        if 'relativisticDynamicalFrictionCorrections' in opt.additionalParameters and opt.additionalParameters['relativisticDynamicalFrictionCorrections']:
            relCovFactor = (1. + v**2)**2 / (1. - v**2)

        if opt.haloPhaseSpaceDescription:
            density = sp.halo.density(r, v_max=(opt.additionalParameters['v_max'] if 'v_max' in opt.additionalParameters else v))
        else:
            density = sp.halo.density(r) * opt.dmPhaseSpaceFraction

        return 4.*np.pi * relCovFactor * sp.m2**2 * density * ln_Lambda / v**2


    def BH_cross_section(sp, r, v, opt=EvolutionOptions()):
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


    def dm2_dt(sp, r, v, opt=EvolutionOptions()):
        """
        The function gives the mass gain due to accretion of the small black hole inside of the dark matter halo
           for a small black hole with relative velocity v to the halo at radius r
        The equation of https://arxiv.org/pdf/1711.09706.pdf is used

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radial position of the black hole in the halo
            v  (float)      : The relative velocity
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The mass gain due to accretion
        """
        return sp.halo.density(r) * v * Classic.BH_cross_section(sp, r, v, opt)


    def dm2_dt_avg(sp, a, e=0., opt=EvolutionOptions()):
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
                dm2_dt = Classic.dm2_dt(sp, a, v_s, opt)
            else:
                if  isinstance(a, (collections.Sequence, np.ndarray)):
                    return np.array([Classic.dm2_dt(sp, a_i, e, opt) for a_i in a])
                def integrand(phi):
                    r = a*(1. - e**2)/(1. + e*np.cos(phi))
                    v_s = np.sqrt(sp.m_total(a) *(2./r - 1./a))
                    return Classic.dm2_dt(sp, r, v_s) / (1.+e*np.cos(phi))**2
                dm2_dt = (1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]

        dm2_baryons_dt = 0.
        if opt.baryonicHaloEffects and opt.baryonicEvolutionOptions.accretion:
            dmHalo = sp.halo
            sp.halo = sp.baryonicHalo
            dm2_baryons_dt = Classic.dm2_dt_avg(sp, a, e, opt.baryonicEvolutionOptions)
            sp.halo = dmHalo

        if(opt.verbose > 2):
            print(f"dm2_dt={dm2_dt}, dm2_baryons_dt = {dm2_baryons_dt}")
        return dm2_dt + dm2_baryons_dt


    def F_acc(sp, r, v, opt=EvolutionOptions()):
        """
        The function gives the force mimicing the accretion effects of an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float)      : The speed of the orbiting object wrt to the dark matter halo
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the accretion force
        """
        return Classic.dm2_dt(sp, r, v, opt) *  v


    def F_acc_recoil(sp, r, v, opt=EvolutionOptions()):
        """
        The function gives the recoil force of the accretion of an object inside a dark matter halo at radius r (since we assume a spherically symmetric halo)
            and with velocity v

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float)      : The speed of the orbiting object wrt to the dark matter halo
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the accretion recoil force
        """
        return Classic.dm2_dt(sp, r, v, opt) * v

    def F_gas(sp, r, v, opt=EvolutionOptions()):
        """
        The function gives the force an accretion disc would exert on a smaller black hole for different models of gas interaction

        Choose model through opt.gasInteraction parameter
            'gasTorqueLoss' : according to https://arxiv.org/pdf/2206.05292.pdf
            'dynamicalFriction' : according to https://arxiv.org/pdf/2006.10206.pdf


        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float)      : The speed of the orbiting object
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the force through gas interactions
        """
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
                alpha = sp.halo.alpha  # requires ShakuraSunyaevDisc atm
            else:
                alpha = opt.additionalParameters['gasTorqueAlpha'] if 'gasTorqueAlpha' in opt.additionalParameters else 0.1

            Omega = sp.omega_s(r)
            Gamma_vis = 3.*np.pi * alpha * Sigma * r**4 * Omega**2 / mach_number**2 if mach_number > 0. else 0.

            fudge_factor = opt.additionalParameters['gasTorqueFudgeFactor'] if 'gasTorqueFudgeFactor' in opt.additionalParameters else 1.
            Gamma_gas = fudge_factor * Gamma_vis

            F_gas = Gamma_gas * sp.m2/sp.m1 / r

        elif opt.additionalParameters['gasInteraction'] == 'dynamicalFriction':
            H = sp.halo.scale_height(r)
            v_rel = v   # TODO: Improve
            R_acc = 2.*sp.m2 /v_rel**2
            Sigma = sp.halo.surface_density(r)
            coulombLog = opt.additionalParameters['gasCoulombLog'] if 'gasCoulombLog' in opt.additionalParameters else 7.15*H/R_acc

            F_gas =  np.sqrt(8.*np.pi) * Sigma * sp.m2**2 * coulombLog / H / v_rel**2 if Sigma > 0. else 0.

        return F_gas



    def dE_force_dt(sp, F, a, e=0., opt=EvolutionOptions()):
        """
        The function calculates the energy loss due to a force F(r,v) by averaging over a Keplerian orbit
           with semimajor axis a and eccentricity e
        According to https://arxiv.org/abs/1908.10241

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            F  (callable(sp, r, v, options)) : The function that represents the force term
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy loss due to accretion
        """
        if e == 0.:
            v_s = sp.omega_s(a)*a
            return - F(sp, a, v_s, opt)*v_s
        else:
            if  isinstance(a, (collections.Sequence, np.ndarray)):
                return np.array([Classic.dE_force_dt(sp, a_i, e, opt) for a_i in a])
            def integrand(phi):
                r = a*(1. - e**2)/(1. + e*np.cos(phi))
                v_s = np.sqrt(sp.m_total(a) *(2./r - 1./a))
                return F(sp, r, v_s, opt)*v_s / (1.+e*np.cos(phi))**2
            return -(1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    def dL_force_dt(sp, F, a, e, opt=EvolutionOptions()):
        """
        The function calculates the angular momentum loss due to a force F(r,v) by averaging over a Keplerian orbit
           with semimajor axis a and eccentricity e
        According to https://arxiv.org/abs/1908.10241

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            F  (callable(sp, r, v, options)) : The function that represents the force term
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum loss due to accretion
        """
        def integrand(phi):
            r = a*(1. - e**2)/(1. + e*np.cos(phi))
            v_s = np.sqrt(sp.m_total(a) *(2./r - 1./a))
            return F(sp, r, v_s, opt) / v_s / (1.+e*np.cos(phi))**2
        return -(1.-e**2)**(3./2.)/2./np.pi *np.sqrt(sp.m_total(a) * a*(1.-e**2)) *  quad(integrand, 0., 2.*np.pi, limit = 100)[0]


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
        dE_gw_dt = Classic.dE_gw_dt(sp, a, e, opt) if opt.gwEmissionLoss else 0.
        dE_df_dt = Classic.dE_force_dt(sp, Classic.F_df, a, e, opt) if opt.dynamicalFrictionLoss else 0.
        dE_acc_dt = Classic.dE_force_dt(sp, Classic.F_acc, a, e, opt) if opt.accretionForceLoss else 0.
        dE_acc_dt += Classic.dE_force_dt(sp, Classic.F_acc_recoil, a, e, opt) if opt.accretionRecoilLoss else 0.
        dE_gas_dt = Classic.dE_force_dt(sp, Classic.F_gas, a, e, opt) if 'gasInteraction' in opt.additionalParameters else 0.

        dE_baryons_dt = 0.
        if opt.baryonicHaloEffects:
            dmHalo = sp.halo
            sp.halo = sp.baryonicHalo
            dE_baryons_dt = Classic.dE_dt(sp, a, e, opt.baryonicEvolutionOptions)
            sp.halo = dmHalo

        if opt.verbose > 2:
            print(f"dE_gw_dt= {dE_gw_dt}, dE_df_dt= {dE_df_dt}, dE_acc_dt= {dE_acc_dt}, dE_gas_dt= {dE_gas_dt}, dE_baryons_dt = {dE_baryons_dt}")
        return ( dE_gw_dt + dE_df_dt + dE_acc_dt + dE_gas_dt + dE_baryons_dt)


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
        dL_gw_dt = Classic.dL_gw_dt(sp, a, e, opt) if opt.gwEmissionLoss else 0.
        dL_df_dt = Classic.dL_force_dt(sp, Classic.F_df, a, e, opt) if opt.dynamicalFrictionLoss else 0.
        dL_acc_dt = Classic.dL_force_dt(sp, Classic.F_acc, a, e, opt) if opt.accretionForceLoss else 0.
        dL_acc_dt += Classic.dL_force_dt(sp, Classic.F_acc_recoil, a, e, opt) if opt.accretionRecoilLoss else 0.
        dL_gas_dt = Classic.dL_force_dt(sp, Classic.F_gas, a, e, opt) if 'gasInteraction' in opt.additionalParameters else 0.

        dL_baryons_dt = 0.
        if opt.baryonicHaloEffects:
            dmHalo = sp.halo
            sp.halo = sp.baryonicHalo
            dL_baryons_dt = Classic.dL_dt(sp, a, e, opt.baryonicEvolutionOptions)
            sp.halo = dmHalo

        if opt.verbose > 2:
            print(f"dL_gw_dt= {dL_gw_dt}, dL_df_dt= {dL_df_dt}, dL_acc_dt= {dL_acc_dt}, dL_gas_dt= {dL_gas_dt}, dL_baryons_dt = {dL_baryons_dt}")
        return  (dL_gw_dt + dL_df_dt + dL_acc_dt + dL_gas_dt + dL_baryons_dt)


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
        accretion = opt.accretion or (opt.baryonicHaloEffects and opt.baryonicEvolutionOptions.accretion)

        def g(e):
            return e**(12./19.)/(1. - e**2) * (1. + 121./304. * e**2)**(870./2299.)

        t_coal =  5./256. * a_0**4/sp.m_total()**2 /sp.m_reduced()
        if opt.elliptic:
            t_coal = t_coal * 48./19. / g(e_0)**4 * quad(lambda e: g(e)**4 *(1-e**2)**(5./2.) /e/(1. + 121./304. * e**2), 0., e_0, limit=100)[0]   # The inspiral time according to Maggiore (2007)

        if t_fin is None:
            t_fin = 1.2 * t_coal *( 1. - a_fin**4 / a_0**4)    # This is the time it takes with just gravitational wave emission

        if a_fin == 0.:
            a_fin = sp.r_isco()     # Stop evolution at r_isco

        a_scale = a_0
        t_scale = t_fin
        m_scale = sp.m2 if accretion else 1.

        t_step_max = np.inf
        if opt.verbose > 0:
            print("Evolving from ", a_0/sp.r_isco(), " to ", a_fin/sp.r_isco(),"r_isco ", ("with initial eccentricity " + str(e_0)) if opt.elliptic else " on circular orbits", " with ", opt)

        # Define the evolution function
        def dy_dt(t, y, *args):
            sp = args[0]; opt = args[1]
            t = t*t_scale

            # Unpack array
            a, e, m2 = y
            a *= a_scale; sp.m2 = m2 * m_scale if accretion else sp.m2

            if opt.verbose > 1:
                tic = time.perf_counter()

            da_dt, dE_dt = Classic.da_dt(sp, a, e, opt=opt, return_dE_dt=True)
            de_dt = Classic.de_dt(sp, a, e, dE_dt=dE_dt, opt=opt) if opt.elliptic else 0.
            dm2_dt = Classic.dm2_dt_avg(sp, a, e, opt) if accretion else 0.

            if opt.verbose > 1:
                toc = time.perf_counter()
                print("t=", t, "a=", a, "da/dt=", da_dt, "e=", e, "de/dt=", de_dt, "m2=", sp.m2, "dm2_dt=", dm2_dt,
                        " elapsed real time: ", toc-tic)

            dy = np.array([da_dt/a_scale, de_dt, dm2_dt/m_scale])
            return dy * t_scale

        # Termination condition
        fin_reached = lambda t,y, *args: y[0] - a_fin/a_scale
        fin_reached.terminal = True

        # Initial conditions
        y_0 = np.array([a_0 / a_scale, e_0, sp.m2/m_scale])

        # Evolve
        tic = time.perf_counter()
        Int = solve_ivp(dy_dt, [t_0/t_scale, (t_0+t_fin)/t_scale], y_0, dense_output=True, args=(sp,opt), events=fin_reached, max_step=t_step_max/t_scale,
                                                                                        method = 'RK45', atol=opt.accuracy, rtol=opt.accuracy)
        toc = time.perf_counter()

        # Collect results
        t = Int.t*t_scale
        a = Int.y[0]*a_scale;
        ev = Classic.EvolutionResults(sp, opt, t, a, msg=Int.message)
        ev.e = Int.y[1] if opt.elliptic else 0.
        ev.m2 = Int.y[2]*m_scale if accretion else sp.m2;

        if opt.verbose > 0:
            print(Int.message)
            print(f" -> Evolution took {toc-tic:.4f}s")

        return ev


class HaloFeedback:
    """
    A class bundling the functions to simulate an inspiral according to the HaloFeedback Model given by https://arxiv.org/abs/2002.12811.pdf
    This class needs an instant, as it calculates a grid for the elliptic function evaluation on the fly

    Attributes:
        N_b (int): The grid size of the impact parameter b
        b_min (float) : The minimum impact parameter b, given by an approximate neutron star size
        sp (SystemProp) : The object describing the information of the inspiral system. sp.halo needs to be of DynamicalSS type
        options (EvolutionOptions) : The object collecting options for the evolution
        m_grid (np.ndarray) : The grid of parameter m on which the elliptic function is calculated
        phi_grid (np.ndarray) : The grid of angle phi on which the elliptic function is calculated
        ell_grid (np.ndarrya) : The result of the elliptic function on the grid
    """
    N_b = 50
    b_min = 15e3 * ms.m_to_pc   # 15 km in pc
    m_grid_density = 6

    class EvolutionOptions:
        """
        This class allows to modify the behavior of the evolution of the differential equations

        Attributes:
            accuracy : float
                An accuracy parameter that is passed to solve_ivp
            verbose : int
                A verbosity parameter ranging from 0 to 2
            gwEmissionLoss : bool
                Whether to include energy losses by graviational waves
            dynamicalFrictionLoss : bool
                Whether to include energy losses by dynamical friction
            haloPhaseSpaceDescription : bool
                Whether to use the phase space description of the halo to calculate relative velocities
                This requires the SystemProp.halo to be of type DynamicSS

        """
        def __init__(self, accuracy=1e-8, verbose=1, gwEmissionLoss=True, dynamicalFrictionLoss=True,
                            baryonicHaloEffects=False, baryonicEvolutionOptions=None,
                            coulombLog=-1, **kwargs):
            self.accuracy = accuracy
            self.verbose = verbose
            self.gwEmissionLoss = gwEmissionLoss
            self.dynamicalFrictionLoss = dynamicalFrictionLoss
            # included for compatibility purposes
            self.accretion = False
            self.accretionForceLoss = False
            self.accretionRecoilLoss = False
            self.haloPhaseSpaceDescription = True
            self.elliptic = False
            self.baryonicHaloEffects = baryonicHaloEffects
            self.baryonicEvolutionOptions = baryonicEvolutionOptions
            self.additionalParameters = kwargs
            self.ln_Lambda = coulombLog
            self.dmPhaseSpaceFraction = 1.

            if not self.baryonicEvolutionOptions is None:
                self.baryonicEvolutionOptions.baryonicHaloEffects = False
                self.baryonicEvolutionOptions.baryonicEvolutionOptions = None
                self.baryonicEvolutionOptions.gwEmissionLoss = False

    def __init__(self, sp, options=EvolutionOptions()):
        """
        The constructor for the HaloFeedback class

        Parameters:
            sp : SystemProp
                The object describing the inspiral system. sp.halo needs to be of DynamicalSS type
        """
        self.sp = sp
        self.m_grid = np.array([])
        self.options=options


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
        if len(m) == 0.:    # can happen depending on mask
            return np.array([])
        phi = np.append(phi1, phi2)
        n_per_decade = self.m_grid_density

        if len(self.m_grid) == 0:   # if the grid is empty it needs to be initialized
            self.m_grid = np.append(0., np.geomspace(1e-5, max(1.001*np.max(m), 1000.), n_per_decade * 8 * max(int(np.log10(np.max(m))), 1)))
            self.phi_grid =  np.append(0., np.geomspace( 1e-5, 1.001*np.pi/2., 6*n_per_decade))
            mphi_grid = np.array(np.meshgrid(self.m_grid, self.phi_grid)).T.reshape(-1,2)
            self.ell_grid = HaloFeedback.elliptic_function(mphi_grid[:,0], mphi_grid[:,1])
            self.mphi_grid_tri = Delaunay(mphi_grid, incremental=True)     # this is the triangulated mphi_grid
            self.interpolator = CloughTocher2DInterpolator(self.mphi_grid_tri, self.ell_grid, fill_value=0.)
        else:
            if np.max(self.m_grid) < np.max(m): # if the grid is insufficient for the m values it needs to be appended
                n_add = max(int(n_per_decade* np.log10(np.max(m) / np.max(self.m_grid))), 2)
                add_to_m_grid = np.geomspace(np.max(self.m_grid) * (1. + 1./float(n_add)) , 1.001*np.max(m), n_add)
                newpoints_grid = np.array(np.meshgrid(add_to_m_grid, self.phi_grid)).T.reshape(-1,2)
                new_ell = HaloFeedback.elliptic_function(newpoints_grid[:,0], newpoints_grid[:,1])
                self.m_grid = np.append(self.m_grid, add_to_m_grid)
                self.ell_grid = np.append(self.ell_grid, new_ell)
                self.mphi_grid_tri.add_points(newpoints_grid)
                self.interpolator = CloughTocher2DInterpolator(self.mphi_grid_tri, self.ell_grid, fill_value=0.)

        ell_interp = self.interpolator(np.stack((np.append(m, m), phi), axis=-1))  # this is basically a call to griddata, but with the triangulation of points precomputed by Delaunay
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

        r_eps = self.sp.m1 / Eps
        #r_eps = self.sp.halo.r_of_Eps(Eps)
        r_cut = self.sp.m1 / (Eps + 0.5* v_cut**2 )
        #r_cut = self.sp.halo.r_of_Eps(Eps + 0.5*v_cut**2)

        alpha1 = np.arccos( np.clip(R/b_star * (1. - R/r_eps),  -1., 1.) )
        alpha2 = np.arccos( np.clip(R/b_star * (1. - R/r_cut), -1., 1.))
        m = 2.* b_star/R / (1. - R/r_eps + b_star/R)

        mask = (Eps > ( self.sp.halo.potential(R) * (1. - b_star/R) - 1./2. * v_cut**2 )) & ( Eps < self.sp.halo.potential(R)*(1. + b_star/R))  & (m > 0.) & (alpha2 > alpha1)

        ''' # Uncomment to test the performance of the elliptic term interpolation
        tic = time.perf_counter()
        ellipticIntegral0 = HaloFeedback.elliptic_function(m[mask], (np.pi-alpha1[mask])/2.) - HaloFeedback.elliptic_function(m[mask], (np.pi-alpha2[mask])/2.)
        toc = time.perf_counter()
        t0 = toc-tic

        tic = time.perf_counter()
        '''
        ellipticIntegral = self.elliptic_term_interp( m[mask], (np.pi-alpha1[mask])/2., (np.pi-alpha2[mask])/2.)
        '''
        toc = time.perf_counter()
        t = toc-tic
        dif = ellipticIntegral - ellipticIntegral0
        print("difference between scipy and interpolated elliptic integrals: t(e) = ", t, "s vs t(e0)=", t0,
                "smax[e-e0]=", np.max(np.abs(dif)), "max[e/e0-1=]", np.max(np.abs(dif/ellipticIntegral)), "(at m=", m[mask][np.argmax(np.abs(dif/ellipticIntegral))], "phi1=", (np.pi - alpha1[mask])[np.argmax(np.abs(dif/ellipticIntegral))]/2.,
                ") avg[e/e0-1=]", np.mean(np.abs(dif/ellipticIntegral)), " for len(m_grid)=", len(self.m_grid))
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
        tic0 = time.perf_counter()
        Eps_grid = self.sp.halo.Eps_grid
        f_grid = self.sp.halo.f_grid
        T_orb = 2.*np.pi / self.sp.omega_s(R)
        v_0 = R* self.sp.omega_s(R)
        b_90 = self.sp.m2/v_0**2
        b_min = self.b_min
        b_max = np.sqrt(self.sp.m1/self.sp.m2) * np.sqrt(b_90**2 + b_min**2)
        b_grid = np.geomspace(b_min, b_max, self.N_b)[::-1] # invert so that DeltaEps is ascending

        b_star, Eps = np.meshgrid( b_grid, Eps_grid)
        DeltaEps =  2.*v_0**2 / (1. + (b_star / b_90)**2)

        tic1 = time.perf_counter()
        P_minus  = self.scatter_probability(R, Eps, DeltaEps, b_star, v_0, b_90, v_cut)

        EpspDeltaEps = Eps + DeltaEps
        P_plus = self.scatter_probability(R, EpspDeltaEps, DeltaEps, b_star, v_0, b_90, v_cut)
        tic2 = time.perf_counter()

        norm = simpson(HaloFeedback.P_DeltaEps(v_0, DeltaEps[0], b_90, b_min, b_max), x=DeltaEps[0])

        '''# Plots the scattering probability on a 2D grid
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection="3d")
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
        #plt.show()
        '''

        dfHalo = np.zeros(np.shape(f_grid))
        # The first term of eq (4.7)
        dfHalo -= f_grid * simpson(P_minus, x=DeltaEps)/T_orb/norm

        # correction calculation to conserve phase space density on a given t_scale, for comparison to Kavanagh code
        correction = np.ones(np.shape(f_grid))
        if not t_scale is None:
            correction = np.clip(f_grid / (-dfHalo * t_scale + 1e-50), 0., 1.)
            dfHalo = np.clip(dfHalo, -f_grid/t_scale, 0.)

        tic3 = time.perf_counter()
        # The second term of eq (4.7)
        dfHalo += simpson(    (Eps/EpspDeltaEps )**(5./2.)
                                #* np.interp(EpspDeltaEps, Eps_grid, f_grid*correction)
                                * griddata(Eps_grid, f_grid*correction, EpspDeltaEps, method='cubic', fill_value=0.)
                                * P_plus
                        , x=DeltaEps) / T_orb/norm

        tic4 = time.perf_counter()
        #print(f"dfHalo_dt: 0->1: {tic1 - tic0}s,  1->2: {tic2-tic1}s, 2->3: {tic3-tic2}s, 3->4: {tic4-tic3}s")
        return dfHalo


    def dR_dt(self, sp, R, v_cut=None):
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
        self.options.v_max = v_cut
        dE_dt = Classic.dE_dt(sp, R, e=0., opt=self.options)
        return (dE_dt)/ Classic.dE_orbit_da(sp, R)

    class EvolutionResults:
        """
        This class keeps track of the evolution of an inspiral.

        Attributes:
            sp : merger_system.SystemProp
                The system properties used in the evolution
            opt : HaloFeedback.EvolutionOptions
                The options used during the evolution
            t : np.ndarray
                The time steps of the evolution
            a,R : np.ndarray
                The corresponding values of the semimajor axis - since e=0, this is also called R
            f   : np.ndarray
                The corresponding values of the distribution function
            e  : float/np.ndarray
                The corresponding values of the eccentricity = 0, for compatibility purposes
            m2 : float/np.ndarray
                The corresponding values of the mass of the secondary object, for compatibility purposes
            msg : string
                The message of the solve_ivp integration
        """
        def __init__(self, sp, options, t, R, f, msg=None):
            self.sp = sp
            self.options = options
            self.msg=msg
            self.t = t
            self.R = R
            self.a = R
            self.f = f
            self.m2 = sp.m2
            self.e = np.zeros(np.shape(t))

        def save(self, filename):
            np.savez(filename, t=self.t, R=self.R, f=self.f)

        def load(filename, sp, options):
            npzf = np.load(filename)
            return HaloFeedback.EvolutionResults(sp, options, npzf['t'], npzf['R'], npzf['f'])

    def Evolve(self, R_0, R_fin=0., t_0=0., t_fin=None):
        """
        The function evolves the system of differential equations as described in https://arxiv.org/pdf/2002.12811.pdf
         for the radius R and for the distribution function f on a grid Eps using the solve_ivp function

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system, sp.halo needs to be of type DynamicSS
            R_0  (float)    : The initial orbital radius
            R_fin (float)   : The orbital radius at which to stop evolution
            t_0    (float)  : The initial time
            t_fin (float)   : The time at which to stop the evolution

        Returns:
            ev : Evolution
                An evolution object that contains the results
        """
        if R_fin <= 0.:
            R_fin = sp.r_isco()

        t_coal =  5./256. * R_0**4/self.sp.m_total()**2 /self.sp.m_reduced() # inspiral time from just GW, as a kind of maximum time
        if t_fin is None:
            t_fin = t_coal *( 1. - R_fin**4 / R_0**4)

        Tfin_orb =  2.*np.pi / self.sp.omega_s(R_fin)
        Tini_orb = 2.*np.pi / self.sp.omega_s(R_0)

        t_scale = 1e3 * Tfin_orb
        R_scale = R_fin

        t_step_max = np.inf

        if self.options.verbose > 0:
            print("Evolving from ", R_0/self.sp.r_isco(), " to ", R_fin/self.sp.r_isco(),"r_isco")
            tic_init = time.perf_counter()

        # Define evolution function
        def dy_dt(t, y):
            R = y[0]*R_scale
            v_0 = self.sp.omega_s(R)*R
            self.sp.halo.f_grid = y[1:]

            if self.options.verbose > 1:
                tic = time.perf_counter()

            dR_dt = self.dR_dt(self.sp, R, v_cut=v_0)
            df_dt = self.dfHalo_dt(R, v_cut=v_0)

            if self.options.verbose > 1:
                toc = time.perf_counter()
                print('t=', t, 'R=', R/self.sp.r_isco(), 'y =', y, 'dR/dt = ', dR_dt*R_scale/self.sp.r_isco(), 'df_dt = ', df_dt, f" elapsed real time {toc-tic:0.4f} seconds")
            return np.append(dR_dt / R_scale, df_dt) * t_scale

        # Termination condition
        fin_reached = lambda t,y, *args: y[0] - R_fin/R_scale
        fin_reached.terminal = True

        # Initial conditions
        y_0 = np.append(R_0/R_scale, self.sp.halo.f_grid)
        y_0_atol = np.append(self.options.accuracy, self.sp.halo.f_grid * np.sqrt(self.options.accuracy))
        y_0_rtol = self.options.accuracy

        # Evolve
        Int = solve_ivp(dy_dt, [t_0/t_scale, (t_0+t_fin)/t_scale], y_0, dense_output=True, events=fin_reached, max_step=t_step_max,
                                                                                        method = 'RK23', atol=y_0_atol, rtol=y_0_rtol)

        # Collect results
        R = Int.y[0]*R_scale
        f = np.transpose(Int.y[1:])
        t = Int.t*t_scale
        ev = HaloFeedback.EvolutionResults(self.sp, self.options, t, R, f, msg=Int.message)

        if self.options.verbose > 0:
            toc = time.perf_counter()
            print(Int.message)
            print(f" -> Evolution took {toc-tic_init}s and {len(t) :,} steps")

        return ev

    def Evolve_HFK(self, R_0, R_fin=0., t_0=0., t_fin=None, dt=None, N_step=None, adjust_stepsize=False, dt_Torb=None):
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
        if (not t_fin is None) and (not dt is None) and (not N_step is None):
            print("You can only specify 2 out of 3 of t_fin, dt, N_step")
            return

        t_coal =  5./256. * R_0**4/self.sp.m_total()**2 /self.sp.m_reduced()
        if t_fin is None and (N_step is None or dt is None):
            t_fin = t_coal *( 1. - R_fin**4 / R_0**4)
        elif t_fin is None:
            t_fin = dt*N_step
        Tfin_orb =  2.*np.pi / self.sp.omega_s(R_fin)
        Tini_orb = 2.*np.pi / self.sp.omega_s(R_0)

        if dt_Torb is None:
            dt_Torb = 5e3
        if dt is None and (not t_fin is None) and (not N_step is None):
            dt = t_fin/float(N_step)
        elif dt is None:
            dt = dt_Torb *Tini_orb

        if N_step is None:
            N_step = int(t_fin/dt)

        if self.options.verbose > 0:
            print("Evolving from ", R_0/self.sp.r_isco(), " to ", R_fin/self.sp.r_isco(),"r_isco", " - dt=", dt, " with max steps ", N_step)

        t_list = np.array([t_0]);   t = t_0
        f_list = np.array([self.sp.halo.f_grid]);
        R_list = np.array([R_0]);   R = R_0

        i = 0
        while i < N_step and R > R_fin and t <= t_fin:
            v_0 = self.sp.omega_s(R)*R

            tic = time.perf_counter()
            df1 = dt * self.dfHalo_dt(R, v_cut=v_0, t_scale=dt)
            dr1 = dt * self.dR_dt(self.sp, R)
            self.sp.halo.f_grid += df1; R += dr1;
            df2 = dt * self.dfHalo_dt(R, v_cut=v_0, t_scale=dt)
            dr2 = dt * self.dR_dt(self.sp, R)
            self.sp.halo.f_grid += 0.5 * (df2-df1);  R += 0.5*(dr2-dr1);
            t += dt
            toc = time.perf_counter()

            print(i, "t=",t, ",dt=", dt, ",R=", R/self.sp.r_isco(), ", dr/dt=", 0.5*(dr1 + dr2), ", f=" , self.sp.halo.f_grid, ", df/ft=", 0.5*(df1+df2) , " elapsed time ", toc-tic)
            t_list = np.append(t_list, t+dt)
            f_list = np.concatenate((f_list, [self.sp.halo.f_grid]))
            R_list = np.append(R_list, R)
            i+= 1
            T_orb = 2.*np.pi/self.sp.omega_s(R)
            if adjust_stepsize:
                dt = np.min([dt, dt_Torb*T_orb])

        # Collect results
        ev = HaloFeedback.EvolutionResults(self.sp, self.options, t_list, R_list, f_list)

        return ev


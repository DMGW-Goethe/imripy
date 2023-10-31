import numpy as np
import imripy.cosmo as cosmo
import imripy.halo as halo
#from scipy.constants import G, c



class SystemProp:
    """
    A class describing the properties of a binary system

    Attributes:
        m1 (float): The mass of the central object, usually a black hole - in units of (solar_mass *G/c**2)
        m2 (float): The mass of the secondary object, usually a neutron star or smaller black hole - in units of (solar_mass *G/c**2)
        D  (float): The luminosity distance to the system
        halo (DMHalo): The object describing the dark matter halo around the central object
    """


    def __init__(self, m1, m2, halo=halo.ConstHalo(0.), D=1., inclination_angle = 0., pericenter_angle=0., baryonicHalo=None, includeHaloInTotalMass=False):
        """
        The constructor for the SystemProp class

        Parameters:
            m1 : float
                The mass of the central object, usually a black hole
            m2 : float
                The mass of the secondary object, usually a neutron star or smaller black hole - in units of (solar_mass *G/c**2)
            halo : DMHalo
                The DMHalo object describing the dark matter halo around the central object
            D :  float
                The luminosity distance to the system
            inclination_angle : float
                The inclination angle (usually denoted iota) at which the system is oriented, see https://arxiv.org/pdf/1807.07163.pdf
            pericenter_angle : float
                The angle at which the pericenter is located wrt the observer, denoted as beta in https://arxiv.org/pdf/1807.07163.pdf
            includeHaloInTotalMass : bool
                Whether to include the dark matter halo mass in the calculation of the enclosed mass
        """
        self.m1 = m1
        self.m2 = m2

        self.D = D

        self.halo = halo
        self.halo.r_min = self.r_isco() if halo.r_min == 0. else halo.r_min

        self.baryonicHalo = baryonicHalo
        if not self.baryonicHalo is None:
            self.baryonicHalo.r_min = self.r_isco()

        self.inclination_angle = inclination_angle
        self.pericenter_angle = pericenter_angle

        self.includeHaloInTotalMass = includeHaloInTotalMass


    def r_isco(self):
        """
        The function returns the radius of the Innermost Stable Circular Orbit (ISCO) of a Schwarzschild black hole with mass m1

        Returns:
            out : float
                The radius of the ISCO
        """
        return 6.*self.m1

    def r_schwarzschild(self):
        """
        The function returns the Schwarzschild radius of a Schwarzschild black hole with mass m1

        Returns:
            out : float
                The Schwarzschild radius
        """
        return 2.*self.m1

    def m_reduced(self, r=0.):
        """
        The function returns the reduced mass of the binary system of m1 and m2
            if r > 0 then the dark matter halo mass is included in the calculation

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float
                The reduced mass
        """
        return np.where(r > 0., self.mass(r)*self.m2 / (self.mass(r) + self.m2),
                                self.m1 * self.m2 / ( self.m1 + self.m2))

    def redshifted_m_reduced(self, r=0.):
        """
        The function returns the redshifted reduced mass of the binary system of m1 and m2
            if r > 0 then the dark matter halo mass is included in the calculation

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float
                The redshifted reduced mass
        """
        return (1. + self.z()) * self.m_reduced(r)

    def m_total(self, r=0.):
        """
        The function returns the total mass of the binary system of m1 and m2
            if r > 0 then the dark matter halo mass is included in the calculation

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass
        Returns:
            out : float
                The total mass
        """
        return np.where( r > 0., self.mass(r) + self.m2,
                                self.m1+ self.m2)

    def m_chirp(self):
        """
        The function returns the chirp mass of the binary system of m1 and m2

        Returns:
            out : float
                The chirp mass
        """
        return self.m_reduced()**(3./5.) * self.m_total()**(2./5.)

    def redshifted_m_chirp(self):
        """
        The function returns the redshifted chirp mass of the binary system of m1 and m2

        Returns:
            out : float
                The redshifted chirp mass
        """
        return (1.+self.z()) * self.m_chirp()

    def z(self):
        """
        The function returns the redshift as a measure of distance to the system
        According to the Hubble Law

        Returns:
            out : float
                The redshift of the system
        """
        return cosmo.HubbleLaw(self.D)

    def mass(self, r):
        """
        The function returns the total mass enclosed in a sphere of radius r.
            This includes the central mass and the mass of the dark matter halo if includeHaloInTotalMass=True

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or array_like (depending on r)
                The enclosed mass
        """
        return np.ones(np.shape(r))*self.m1 + (self.halo.mass(r) if self.includeHaloInTotalMass else 0.)

    def dmass_dr(self, r):
        """
        The function returns the derivative of the total mass enclosed in a sphere of radius r.
            This derivative stems from the mass of the dark matter halo and is only nonzero if includeHaloInTotalMass=True

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass derivative

        Returns:
            out : float or array_like (depending on r)
                The enclosed mass derivative
        """
        return 4.*np.pi*r**2 * self.halo.density(r) if self.includeHaloInTotalMass else 0.


    def omega_s(self, r):
        """
        The function returns the angular frequency of the smaller mass m2 in a circular orbit around the central mass with the dark matter halo around it

        Parameters:
            r : float or array_like
                The radius at which to evaluate the orbital frequency

        Returns:
            out : float or array_like (depending on r)
                The orbital frequency
        """
        return np.sqrt((self.mass(r) + self.m2)/r**3)


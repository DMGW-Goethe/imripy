import numpy as np
import imripy.cosmo as cosmo
import imripy.halo as halo
#from scipy.constants import G, c

class HostSystem:
    """
    Defines the host system of the central MBH m1 with a halo of stuff around.
    Defines the fundamental plane of the system, to which the inclination angle, pericenter angle and luminosity distance are wrt to an observer

    TODO: Implement handling of the position of the fundamental plane to the observer
    """
    def __init__(self, m1, halo=halo.ConstHalo(0.), D_l = 1., inclination_angle=0., pericenter_angle=0., includeHaloInTotalMass=False):
        self.m1 = m1
        self.D_l = D_l

        self.halo = halo
        self.halo.r_min = self.r_isco() if halo.r_min == 0. else halo.r_min

        self.inclination_angle = inclination_angle
        self.pericenter_angle = pericenter_angle

        self.includeHaloInTotalMass = includeHaloInTotalMass

    @property
    def r_isco(self):
        """
        The function returns the radius of the Innermost Stable Circular Orbit (ISCO) of a Schwarzschild black hole with mass m1

        Returns:
            out : float
                The radius of the ISCO
        """
        return 6.*self.m1

    @property
    def r_schwarzschild(self):
        """
        The function returns the Schwarzschild radius of a Schwarzschild black hole with mass m1

        Returns:
            out : float
                The Schwarzschild radius
        """
        return 2.*self.m1

    @property
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
            This includes the central mass and the mass of the matter halo if includeHaloInTotalMass=True

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
            This derivative stems from the mass of the matter halo and is only nonzero if includeHaloInTotalMass=True

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
        The function returns the angular frequency of a test mass in a circular orbit around the central mass with the matter halo around it

        Parameters:
            r : float or array_like
                The radius at which to evaluate the orbital frequency

        Returns:
            out : float or array_like (depending on r)
                The orbital frequency
        """
        return np.sqrt(self.mass(r)/r**3)


class KeplerOrbit:
    """
    Parameterizes a Keplerian orbit of a secondary with mass m2 wrt to the fundamental plane of the central MBH m1
    The Keplerian orbital parameters are given by
        a (float)   : semi-major axis of the ellipses
        e (float)   : eccentricity (0<= e < 1) of the ellipses
        inclination_angle (float)   : The inclination of the plane of the orbit wrt the fundamental plane measured at the ascending note
        longitude_an      (float)   : The angle of the ascending node wrt to the reference direction of the fundamental plane
        periapse_angle    (float)   : The angle of the point of periapse wrt to the ascending node

    We also define whether the true anomaly has the same orientation as angular momentum in the fundamental plane with the parameter
        prograde (Boolean) : Whether the secondary is prograde wrt to some direction in the fundamental plane
    """

    def __init__(self, hs, m2, a, e=0., periapse_angle=0., inclination_angle=0., longitude_an=0., prograde=True):
        """
        The constructor for the KeplerOrbit class

        Parameters
        ---------
            hs (HostSystem)
                The host system object
            m2 (float)
                The mass of the secondary
            a (float)
                semi-major axis
            e (float)
                eccentricity (0<= e < 1)
            inclination_angle (float)
                The inclination angle
            longitude_an (float)
                The angle of the ascending node
            periapse_angle    (float)
                The angle of the point of periapse
            prograde (Boolean)
                Whether the secondary is prograde wrt to some direction in the fundamental plane
        """
        self.hs = hs
        self.m2 = m2
        self.a = a
        self.e = e
        self.periapse_angle = periapse_angle
        self.inclination_angle = inclination_angle
        self.longitude_an = longitude_an
        self.prograde = prograde

    def from_xy_plane_to_rhophi_plane(self, v):
        """
        Relates a point in an (x,y,z) plane to a point in an (rho,phi,z) plane

        Parameters
        ----------
            v : np.ndarry
                3 dimensional vector with (x,y,z)

        Returns
            np.ndarray
                3 dimenstional vector with corresponding (rho, phi, z)
        """
        x = v[0]; y = v[1]
        rho = np.sqrt(x**2 + y**2)
        phi = np.sign(x) * np.arcsin(y/rho) + ( np.pi if x<0 else 0.)
        return np.array([rho, phi, v[2]])

    def from_rhophi_plane_to_xy_plane(self, v):
        """
        Relates a point in an (rho, phi, z) plane to a point in an (r,x,z) plane

        Parameters
        ----------
            v : np.ndarry
                3 dimensional vector with (rho,phi,z)

        Returns
            np.ndarray
                3 dimenstional vector with corresponding (x,y, z)
        """
        rho = v[0]; phi = v[1]
        return np.array([rho*np.cos(phi), rho*np.sin(phi), v[2]])

    def from_orbital_xy_plane_to_fundamental_xy_plane(self, x):
        """
        Relates a point in the orbital xyz plane to the fundamental XYZ plane
        See e.g. Gravity - Newtonian, Post-Newtonian, Relativistic by Poisson, Will pg.153

        Parameters
        ----------
            x : np.ndarry
                3 dimensional vector with (x,y,z)

        Returns
            np.ndarray
                3 dimenstional vector with corresponding (X,Y,Z)
        """
        R1 = np.array([[np.cos(self.periapse_angle), -np.sin(self.periapse_angle), 0.],
                        [np.sin(self.periapse_angle), np.cos(self.periapse_angle), 0.],
                        [0., 0., 1.]])
        R2 = np.array([[1., 0., 0.],
                        [0., np.cos(self.inclination_angle), -np.sin(self.inclination_angle)],
                        [0., np.sin(self.inclination_angle), np.cos(self.inclination_angle)] ])
        R3 = np.array([[np.cos(self.longitude_an), -np.sin(self.longitude_an), 0.],
                        [np.sin(self.longitude_an), np.cos(self.longitude_an), 0.],
                        [0., 0., 1.]])
        R = np.matmul(np.matmul(R3, R2), R1)
        return np.matmul(R, x)

    def from_fundamental_xy_plane_to_orbital_xy_plane(self, x):
        """
        Relates a point in the fundamental XYZ plane to the orbital xyz plane

        Parameters
        ----------
            x : np.ndarry
                3 dimensional vector with (X,Y,Z)

        Returns
            np.ndarray
                3 dimenstional vector with corresponding (x,y,z)
        """
        R1 = np.array([[np.cos(self.periapse_angle), -np.sin(self.periapse_angle), 0.],
                        [np.sin(self.periapse_angle), np.cos(self.periapse_angle), 0.],
                        [0., 0., 1.]])
        R2 = np.array([[1., 0., 0.],
                        [0., np.cos(self.inclination_angle), -np.sin(self.inclination_angle)],
                        [0., np.sin(self.inclination_angle), np.cos(self.inclination_angle)] ])
        R3 = np.array([[np.cos(self.longitude_an), -np.sin(self.longitude_an), 0.],
                        [np.sin(self.longitude_an), np.cos(self.longitude_an), 0.],
                        [0., 0., 1.]])
        R = np.matmul(np.matmul(R3, R2), R1)
        return np.matmul(R.T, x) # transpose gives inverse

    def get_orbital_vectors_in_orbital_xy_plane(self, phi):
        """
        Returns the Gaussian decomposition of the orbital vectors
        n points to the secondary, m orthogonal to that inside the orbital plane, k perpendicular to the plane
        in the coordinate system of the orbital xyz plane

        Parameters
        ----------
            phi : float
                The true anomaly at the point of interest throughout the orbit

        Returns
            (np.ndarray, np.ndarray, np.ndarray)
                3  3d vectors n,m,k
        """
        n = np.array([np.cos(phi), np.sin(phi),0.]) # The orbital vectors in the orbital xy plane
        m = np.array([-np.sin(phi), np.cos(phi),0.])
        k = np.array([0.,0.,1.])
        return n,m,k

    def get_orbital_vectors_in_fundamental_xy_plane(self, phi):
        """
        Returns the Gaussian decomposition of the orbital vectors
        n points to the secondary, m orthogonal to that inside the orbital plane, k perpendicular to the plane
        in the coordinate system of the fundamental XYZ plane

        Parameters
        ----------
            phi : float
                The true anomaly at the point of interest throughout the orbit

        Returns
            (np.ndarray, np.ndarray, np.ndarray)
                3  3d vectors n,m,k
        """
        n,m,k = self.get_orbital_vectors_in_orbital_xy_plane(phi)
        n = self.from_orbital_xy_plane_to_fundamental_xy_plane(n)
        m = self.from_orbital_xy_plane_to_fundamental_xy_plane(m)
        k = self.from_orbital_xy_plane_to_fundamental_xy_plane(k)
        return n,m,k

    ''' # Alternatively
    def get_orbital_vectors_in_fundamental_xy_plane(self, phi):
        n = np.array([np.cos(self.longitude_an)*np.cos(self.periapse_angle + phi) - np.cos(self.inclination_angle)*np.sin(self.longitude_an)*np.sin(self.periapse_angle+phi),
                      np.sin(self.longitude_an)*np.cos(self.periapse_angle + phi) + np.cos(self.inclination_angle)*np.cos(self.longitude_an)*np.sin(self.periapse_angle+phi),
                      np.sin(self.inclination_angle)*np.sin(self.periapse_angle+phi)])

        m = np.array([-np.cos(self.longitude_an)*np.sin(self.periapse_angle + phi) - np.cos(self.inclination_angle)*np.sin(self.longitude_an)*np.cos(self.periapse_angle+phi),
                      -np.sin(self.longitude_an)*np.sin(self.periapse_angle + phi) + np.cos(self.inclination_angle)*np.cos(self.longitude_an)*np.cos(self.periapse_angle+phi),
                      np.sin(self.inclination_angle)*np.cos(self.periapse_angle+phi)])

        k = np.array([np.sin(self.inclination_angle)*np.sin(self.longitude_an),
                      -np.sin(self.inclination_angle)*np.cos(self.longitude_an),
                      np.cos(self.inclination_angle)])
        return n,m,k
    '''

    def get_orbital_parameters(self, phi):
        """
        Calculates the orbital position and velocity for the secondary given the true anomaly
            in the fundamental XYZ frame

        Parameters
        ---------
            phi : float
                The true anomaly at the point of interest throughout the orbit

        Returns
        ------
            r (float)                  : The distance of the secondary to the center
            [X, Y, Z] (np.array)       : The position of the secondary
            v (float)                  : The total velocity
            [v_X, v_Y, v_Z] (np.array) : The velocity in XYZ direction
        """
        r = self.a*(1. - self.e**2)/(1. + self.e*np.cos(phi))
        pos_in_orbital_xy_plane = r * np.array([np.cos(phi), np.sin(phi), 0.])

        pos_in_fundamental_xy_plane = self.from_orbital_xy_plane_to_fundamental_xy_plane(pos_in_orbital_xy_plane)

        v = np.sqrt(self.sp.m_total(self.a) *(2./r - 1./self.a))
        v_phi = r * np.sqrt(self.sp.m_total(self.a)*self.a*(1.-self.e**2))/r**2
        v_r = np.sqrt(np.max([v**2 - v_phi**2, 0.]))
        v_in_orbital_xy_plane = np.array([v_r*np.cos(phi) - v_phi*np.sin(phi), v_r*np.sin(phi) + v_phi*np.cos(phi), 0.])
        v_in_fundamental_xy_plane = self.from_orbital_xy_plane_to_fundamental_xy_plane(v_in_orbital_xy_plane)

        return r, pos_in_fundamental_xy_plane, v, v_in_fundamental_xy_plane

    @property
    def m_red(self):
        """
        The function returns the reduced mass of the binary system of m1 and m2

        Returns:
            out : float
                The reduced mass
        """
        return  self.m1 * self.m2 / ( self.m1 + self.m2)

    @property
    def redshifted_m_red(self):
        """
        The function returns the redshifted reduced mass of the binary system of m1 and m2

        Returns:
            out : float
                The redshifted reduced mass
        """
        return (1. + self.hs.z) * self.m_red

    @property
    def m1(self):
        return self.hs.m1

    @property
    def m_tot(self):
        """
        The function returns the total mass of the binary system of m1 and m2

        Returns:
            out : float
                The total mass
        """
        return  self.m1 + self.m2

    @property
    def m_chirp(self):
        """
        The function returns the chirp mass of the binary system of m1 and m2

        Returns:
            out : float
                The chirp mass
        """
        return self.m_red**(3./5.) * self.m_tot**(2./5.)


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
            This includes the central mass and the mass of the matter halo if includeHaloInTotalMass=True

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
            This derivative stems from the mass of the matter halo and is only nonzero if includeHaloInTotalMass=True

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
        The function returns the angular frequency of a test mass in a circular orbit around the central mass with the matter halo around it

        Parameters:
            r : float or array_like
                The radius at which to evaluate the orbital frequency

        Returns:
            out : float or array_like (depending on r)
                The orbital frequency
        """
        return np.sqrt(self.mass(r)/r**3)



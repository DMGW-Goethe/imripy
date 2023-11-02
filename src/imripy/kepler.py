
import numpy as np

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

    @staticmethod
    def from_xy_plane_to_rhophi_plane(v):
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

    @staticmethod
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

    '''
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
    '''

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

    def get_orbital_vectors(self, phi):
        """
        Calculates the orbital position and velocity for the secondary given the true anomaly
            in the fundamental XYZ frame

        Parameters
        ---------
            phi : float
                The true anomaly at the point of interest throughout the orbit

        Returns
        ------
            [X, Y, Z] (np.array)       : The position of the secondary
            [v_X, v_Y, v_Z] (np.array) : The velocity in XYZ direction
        """
        r = self.a*(1. - self.e**2)/(1. + self.e*np.cos(phi))
        pos_in_orbital_xy_plane = r * np.array([np.cos(phi), np.sin(phi), 0.])

        pos_in_fundamental_xy_plane = self.from_orbital_xy_plane_to_fundamental_xy_plane(pos_in_orbital_xy_plane)

        v = np.sqrt(self.m_tot *(2./r - 1./self.a))
        v_phi = r * np.sqrt(self.m_tot*self.a*(1.-self.e**2))/r**2
        v_r = np.sqrt(np.max([v**2 - v_phi**2, 0.]))
        v_in_orbital_xy_plane = np.array([v_r*np.cos(phi) - v_phi*np.sin(phi), v_r*np.sin(phi) + v_phi*np.cos(phi), 0.])
        v_in_fundamental_xy_plane = self.from_orbital_xy_plane_to_fundamental_xy_plane(v_in_orbital_xy_plane)
        v_in_fundamental_xy_plane = v_in_fundamental_xy_plane if self.prograde else -v_in_fundamental_xy_plane

        return pos_in_fundamental_xy_plane, v_in_fundamental_xy_plane

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
            r (float) : The radius of the secondary
            v (float) : The total velocity
        """
        r = self.a*(1. - self.e**2)/(1. + self.e*np.cos(phi))
        v = np.sqrt(self.m_tot *(2./r - 1./self.a))

        return r, v

    @property
    def T(self):
        """
        The orbital period T

        Returns
        -------
            The orbital period of the Keplerian orbit (in pc)
        """
        return 2.*np.pi * np.sqrt(self.a**3 / self.m_tot)

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

    @property
    def redshifted_m_chirp(self):
        """
        The function returns the redshifted chirp mass of the binary system of m1 and m2

        Returns:
            out : float
                The redshifted chirp mass
        """
        return (1.+ self.hs.z) * self.m_chirp


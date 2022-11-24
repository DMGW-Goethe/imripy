from .halo import *

from imripy import constants as c

class NFW(MatterHalo):
    """
    A class describing a Navarro-Frenk-White (NFW) halo profile.
    The density is given by
        rho (r) = rho_0 / (r/r_s) / (1 + r/r_s)**2

    Attributes:
        r_min (float): An minimum radius below which the density is always 0
        rho_0 (float): The density parameter of the profile
        r_s   (float): The scale radius of the profile
    """

    def __init__(self, rho_s, r_s):
        """
        The constructor for the NFW class

        Parameters:
            rho_s : float
                The density parameter of the NFW profile
            r_s : float
                The scale radius of the NFW profile
        """
        MatterHalo.__init__(self)
        self.rho_s = rho_s
        self.r_s = r_s

    def density(self, r):
        """
        The density function of the NFW halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        return np.where(r > self.r_min, self.rho_s / (r/self.r_s) / (1. + r/self.r_s)**2, 0.)


    def mass(self, r):
        """
        The mass that is contained in the NFW halo in the spherical shell of size r.

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or np.ndarray (depending on r)
                The mass inside the spherical shell of size r
        """
        def NFWmass(r):
            return 4.*np.pi*self.rho_s * self.r_s**3 * (np.log((self.r_s + r)/self.r_s) + self.r_s / (self.r_s + r) - 1.)

        return np.where(r > self.r_min,
                            NFWmass(r) - NFWmass(self.r_min),
                             0. )

    def FromHaloMass(M_vir, z_f):
        """
        This function calculates the NFW parameters given by the prescription in II.A of arxiv.org/abs/1408.3534.pdf.

        Parameters:
            M_vir : float
                The virialized mass of the halo in pc (= solar_mass *G/c**2)
            z_f: float
                The formation redshift of the halo

        Returns:
            out : NFW object
                The NFW object with the corresponding rho_0, r_s
                With units [rho_0] = c**2/G/pc^2, [r_s] = pc
        """
        A_200 = 5.71; B_200 = -0.084; C_200 = -0.47; M_piv = 1./0.7 *1e14* 4.8e-14 # solar mass to pc
        c_200 = A_200 * (M_vir/M_piv)**B_200 * (1. + z_f)**C_200
        rho_crit_0 = 3*cosmo.hubble_const**2 / 8./np.pi
        Omega_m = cosmo.Omega_m(z_f)
        rho_m = cosmo.Omega_0_m*rho_crit_0 * (1.+z_f)**3
        Delta_vir = 18.*np.pi**2 *(1. + 0.4093 * (1/Omega_m - 1.)**0.9052)
        r_vir = (3.*M_vir / (4.*np.pi * Delta_vir * rho_m))**(1./3.)
        r_s = r_vir/c_200
        f = np.log(1+c_200) - c_200/(1.+c_200)
        rho_s = 1/3./f * Delta_vir * rho_m * c_200**3
        #print("Halo parameters: ","r_s=", r_s,"rho_s ", rho_s * 1.414e-9) # 1/pc^2 to g/cm^3
        return NFW(rho_s, r_s)

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return f"NFW(rho_s={self.rho_s:0.1e}, r_s={self.r_s:0.1e})"

class Spike(MatterHalo):
    """
    A class describing a spike halo profile
    The density is given by
        rho (r) = rho_spike * (r_spike/r)**(alpha)

    Attributes:
        r_min (float): An minimum radius below which the density is always 0
        rho_0 (float): The density parameter of the spike profile
        r_spike   (float): The scale radius of the spike profile
        alpha     (float): The power-law index of the spike profile, with condition 0 < alpha < 3
    """

    def __init__(self, rho_spike, r_spike, alpha):
        """
        The constructor for the Spike class

        Parameters:
            rho_spike : float
                The density parameter of the spike profile
            r_spike : float
                The scale radius of the spike profile
            alpha : float
                The power-law index of the spike profile, with condition 0 < alpha < 3
        """
        MatterHalo.__init__(self)
        self.rho_spike = rho_spike
        self.alpha= alpha
        self.r_spike = r_spike

    def density(self, r):
        """
        The density function of the spike halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        return np.where(r > self.r_min, self.rho_spike * (self.r_spike/r)**self.alpha, 0.)

    def mass(self, r):
        """
        The mass that is contained in the spike halo in the spherical shell of size r.

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or np.ndarray (depending on r)
                The mass inside the spherical shell of size r
        """
        def spikeMass(r):
            return 4*np.pi*self.rho_spike*self.r_spike**self.alpha * r**(3.-self.alpha) / (3.-self.alpha)

        return np.where(r > self.r_min,
                            spikeMass(r) - spikeMass(self.r_min),
                            0.)

    def FromSpikedNFW(snfw):
        """
        This function extracts the spike parameters from the SpikedNFW class

        Parameters:
            nsfw : SpikedNFW object

        Returns:
            out : Spike object
                A spike object with the corresponding spike parameters
        """
        return Spike(snfw.rho_spike, snfw.r_spike, snfw.alpha)


    def FromRho6(rho_6, M_bh, alpha, r_6=1e-6):
        """
        This function allows to create a new Spike object from the parametrization given in
            https://arxiv.org/pdf/2108.04154.pdf
        where
            rho (r) = rho_6 * (r_6/r)**(alpha)
        and
            r_spike = (  (3-alpha) * 0.2**(3-alpha) * M_bh / 2 / pi / rho_spike )**(1/3)

        Parameters:
            rho_6 : float
                The density at the reference point r_6
            M_bh  : float
                The mass of the central black hole
            alpha : float
                The power-law index of the spike profile, with condition 0 < alpha < 3
            r_6   : float
                The reference point, which is 1e-6 pc by default

        Returns:
            out : Spike object
                A spike object with the corresponding spike parameters
        """
        k = (3. - alpha) * 0.2**(3. - alpha)/2./np.pi
        rho_spike = ( rho_6 * (k*M_bh)**(-alpha/3.) * r_6**alpha )**(3./(3.-alpha))
        r_spike = (k*M_bh/rho_spike)**(1./3.)
        return Spike(rho_spike, r_spike, alpha)


    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return f"Spike(rho_spike={self.rho_spike:0.1e}, r_spike={self.r_spike:0.1e}, alpha={self.alpha:0.1e})"

class RelativisticSpike(MatterHalo):
    """
    A class describing a spike halo profile with relatitivistic corrections
        a la Speeney (2022) https://arxiv.org/pdf/2204.12508.pdf
    The density is given by
        rho (r) = rho_tilde * 10**delta ( rho_0/0.3GeV/cm^3 )**alpha ( M_BH / 1e6 M_sun )**beta ( a / 20kpc )**gamma
        with rho_tilde = A ( 1 - 4 eta / x )**w  ( 4.17e11 / x )**q
        and x = r / M_BH

    Attributes:
        r_min (float): An minimum radius below which the density is always 0
        rho_0 (float): The density parameter of the origin distribution
        M_BH  (float): The mass of the black hole
        a     (float): The scaling parameter of the origin distribution
        alpha (float): The power law index for the density parameter scaling
        beta  (float): The power law index for the BH mass scaling
        gamma (float): The power law index for the scaling parameter scaling
        delta (float): The power law index for the scale of rho
        eta   (float): A factor for either relativistic or newtonian scaling (1 vs 2)
        A     (float): The density fit parameter for rho_tilde
        w     (float): The fit parameter for the first term in rho_tilde
        q     (float): The fit parameter for the second term in rho_tilde
    """

    def __init__(self, M_BH, rho_0, a, alpha, beta, gamma, delta, A, w, q, eta = 1):
        """
        The constructor for the Spike class

        Parameters:
            M_BH  (float)
                The mass of the black hole
            rho_0 (float)
                The density parameter of the origin distribution
            a     (float)
                The scaling parameter of the origin distribution
            alpha (float)
                The power law index for the density parameter scaling
            beta  (float)
                The power law index for the BH mass scaling
            gamma (float)
                The power law index for the scaling parameter scaling
            delta (float)
                The power law index for the scale of rho
            A     (float):
                The density fit parameter for rho_tilde (in units 1/pc^2)
            w     (float)
                The fit parameter for the first term in rho_tilde
            q     (float)
                The fit parameter for the second term in rho_tilde
            eta   (float)
                A factor for either relativistic or newtonian scaling (1 vs 2)
        """
        MatterHalo.__init__(self)
        self.rho_0 = rho_0
        self.M_BH = M_BH
        self.a = a
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.A = A
        self.w = w
        self.q = q
        self.eta = eta
        self.r_min = 4.*self.eta*self.M_BH

    def density(self, r):
        """
        The density function of the realtivistic spike halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        x = r / self.M_BH
        rho_tilde = self.A * (1. - 4*self.eta / x)**self.w  * (4.17e11 / x)**self.q
        rho = rho_tilde * ( 10.**self.delta * (self.rho_0 / 0.3 / c.GeV_cm3_to_invpc2 )**self.alpha
                            * (self.M_BH / 1e6 / c.solar_mass_to_pc)**self.beta * (self.a / 20e3 )**self.gamma )
        return np.where(r > self.r_min, rho, 0.)


    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return f"RelativistcSpike(rho_0={self.rho_0:0.1e}, M_BH={self.M_BH:0.1e}, a={self.a:0.1e}, A={self.A:0.1e}, w={self.w:0.1e}, q={self.q:0.1e})"


class SpikedNFW(NFW, Spike):
    """
    A class describing a Navarro-Frenk-White (NFW) halo profile with a spike below a given radius r_spike.
    The density is given by
        rho (r) = rho_0 / (r/r_s) / (1 + r/r_s)**2    for r > r_spike
        rho (r) = rho_spike * (r_spike/r)**(alpha)    for r < r_spike

    with a continuity condition rho_spike = rho_0 / (r_spike/r_s) / (1 + r_spike/r_s)**2

    Attributes:
        r_min (float): An minimum radius below which the density is always 0
        rho_0 (float): The density parameter of the NFW profile
        r_s   (float): The scale radius of the NFW profile
        r_spike   (float): The scale radius of the spike profile
        rho_spike (float): The density parameter of the spike profile
        alpha     (float): The power-law index of the spike profile, with condition 0 < alpha < 3
    """

    def __init__(self, rho_s, r_s, r_spike, alpha):
        """
        The constructor for the SpikedNFW class

        Parameters:
            rho_0 : float
                The density parameter of the NFW profile
            r_s : float
                The scale radius of the NFW profile
            r_spike : float
                The scale radius of the spike profile
            alpha : float
                The power-law index of the spike profile, with condition 0 < alpha < 3
        """
        NFW.__init__(self, rho_s, r_s)
        rho_spike = rho_s * r_s/r_spike / (1+r_spike/r_s)**2
        Spike.__init__(self, rho_spike, r_spike, alpha)

    def density(self, r):
        """
        The density function of the NFW+spike halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        return np.where(r < self.r_spike,
                        Spike.density(self, r),
                        NFW.density(self, r))

    def mass(self, r):
        """
        The mass that is contained in the NFW+spike halo in the spherical shell of size r.

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or np.ndarray (depending on r)
                The mass inside the spherical shell of size r
        """
        return np.where(r < self.r_spike,
                            Spike.mass(self, r),
                            NFW.mass(self, r) - NFW.mass(self, self.r_spike) + Spike.mass(self, self.r_spike) )


    def FromNFW(nfw, M_bh, alpha):
        """
        This function takes an NFW object and computes the corresponding SpikedNFW profile, given the size of a massive Black Hole at the center,
            according to the description in II.B of https://arxiv.org/pdf/1408.3534.pdf

        Parameters:
            nfw : NFW object
                The NFW object in question
            M_bh : float
                The mass of the Massive Black Hole in the center
            alpha : float
                The power-law index of the spike profile, with condition 0 < alpha < 3

        Returns:
            out : SpikedNFW object
                The SpikedNFW object with the corresponding parameters
        """
        r = np.geomspace(1e-3*nfw.r_s, 1e3*nfw.r_s) # TODO: make this more general
        M_to_r = interp1d(nfw.mass(r), r, kind='cubic', bounds_error=True)
        r_h = M_to_r(2.* M_bh)
        r_spike = 0.2*r_h
        return SpikedNFW(nfw.rho_s, nfw.r_s, r_spike, alpha)

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return f"SpikedNFW(rho_s={self.rho_s:0.1e}, r_s={self.r_s}, r_spike={self.r_spike:0.1e}, alpha={self.alpha:0.1e})"


class Hernquist(MatterHalo):
    """
    A class describing a Hernquist halo profile
    The density is given by
        rho (r) = rho_s / (r/r_s) / (1 + r/r_s)**3

    Attributes:
        r_min (float): An minimum radius below which the density is always 0
        rho_s (float): The density parameter of the Hernquist profile
        r_s   (float): The scale radius of the Hernquist profile
    """
    def __init__(self, rho_s, r_s):
        """
        The constructor for the Hernquist class

        Parameters:
            rho_0 : float
                The density parameter of the Hernquist profile
            r_s : float
                The scale radius of the Hernquist profile
        """
        MatterHalo.__init__(self)
        self.rho_s = rho_s
        self.r_s = r_s

    def density(self, r):
        """
        The density function of the Hernquist halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        return np.where(r > self.r_min, self.rho_s / (r/self.r_s) / (1. + r/self.r_s)**3, 0.)

    def mass(self, r):
        """
        The mass that is contained in the Hernquist halo in the spherical shell of size r.

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or np.ndarray (depending on r)
                The mass inside the spherical shell of size r
        """
        def hernquistMass(r):
            return -4.*np.pi*self.rho_s * self.r_s**3 * (2.*r/self.r_s + 1.)/2./(1.+r/self.r_s)**2


        return np.where(r > self.r_min,
                                hernquistMass(r) - hernquistMass(self.r_min),
                                 0. )

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return f"Hernquist(rho_s={self.rho_s:0.1e}, r_s={self.r_s:0.1e})"


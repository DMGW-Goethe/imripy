import numpy as np
from scipy.interpolate import interp1d, griddata
from scipy.integrate import odeint, quad, simps, solve_ivp
from scipy.optimize import root_scalar
from scipy.special import gamma
#import matplotlib.pyplot as plt
import collections
import imripy.cosmo as cosmo


class MatterHalo:
    """
    A class describing a spherically symmetric, static Matter Halo

    Attributes:
        r_min (float): An minimum radius below which the density is always 0, this is initialized to 0
    """

    def __init__(self):
        """
        The constructor for the MatterHalo class
        """
        self.r_min = 0.

    def density(self, r):
        """
        The density function of the halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        pass

    def mass(self, r, **kwargs):
        """
        The mass that is contained in the halo in the spherical shell of size r.
        This function numerically integrates over the density function, so that it can be used
            in inherited classes that have not implemented an analytic expression for the mass.

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or np.ndarray (depending on r)
                The mass inside the spherical shell of size r
        """
        integrand = lambda r, m: self.density(r, **kwargs)*r**2
        if isinstance(r, (collections.Sequence, np.ndarray)):
            return 4.*np.pi*odeint(integrand, quad(integrand, 0., r[0], args=(0.))[0], r, tfirst=True, rtol=1e-10, atol=1e-10)[:,0]
        else:
            return 4.*np.pi*quad(integrand, 0., r, args=(0.))[0]


    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return "MatterHalo"


class MiyamotoNagaiDisc(MatterHalo):
    """
    A class describing an axisymmetric, static, baryonic disc

    Attributes:
        r_min (float): A minimum radius below which the density is always 0, this is initialized to 0
        M_d   (float): The disc mass paramter
        R_d   (float): The disc scale length
        z_d   (float): The disc scale height
    """

    def __init__(self, M_d, R_d, z_d):
        """
        The constructor for the MiyamotoNagaiDisc class

        Parameters:
            M_d : float
                The disc mass paramter
            R_d : float
                The disc scale length
            z_d : float
                The disc scale height
        """
        self.M_d = M_d
        self.R_d = R_d
        self.z_d = z_d

    def density(self, r, z=0.):
        """
        The density function of the disc, see eq (2.69) of Binney&Tremain(2007)

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density
            z : float or array_like (optional)
                The height at which to evaluate the density (in cylindrical coordinates)

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r, height z
        """
        chi = np.sqrt(z**2 + self.z_d**2)
        return ( self.M_d * self.z_d**2 / 4. / np.pi
                    * (self.R_d* r**2 + (self.R_d + 3.*chi)*(self.R_d + chi)**2 )
                    / ( r**2 + (self.R_d + chi)**2 )**(5./2)  / chi**3 )

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return f"MiyamotoNagaiDisc: M_d={self.M_d}, R_d={self.R_d}, z_d={self.z_d}"

class ConstHalo(MatterHalo):
    """
    A class describing a spherically symmetric, static, and constant Matter Halo

    Attributes:
        r_min (float): An minimum radius below which the density is always 0, this is initialized to 0
        rho_0 (float): The constant density of the halo
    """

    def __init__(self, rho_0):
        """
        The constructor for the ConstHalo class

        Parameters:
            rho_0 : float
                The constant density of the halo
        """
        MatterHalo.__init__(self)
        self.rho_0 = rho_0

    def density(self, r):
        """
        The constant density function of the halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        return np.where(r > self.r_min, self.rho_0, 0.)


    def mass(self, r):
        """
        The mass that is contained in the halo in the spherical shell of size r.

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or np.ndarray (depending on r)
                The mass inside the spherical shell of size r
        """
        return 4./3.*np.pi *self.rho_0 * (r**3 - self.r_min**3)

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return "ConstHalo"


class InterpolatedHalo(MatterHalo):
    """
    A class describing a spherically symmetric, static Matter Halo with the help of an interpolation table.
        Outside of the grid, the density function will return 0.

    Attributes:
        r_min (float): An minimum radius below which the density is always 0, this is initialized to 0
        r_grid (array_like) : The grid in radii
        density_grid (array_like) : The corresponding densities
    """

    def __init__(self, r_grid, density_grid):
        """
        The constructor for the ConstHalo class

        Parameters:
            r_grid : array_like
                The grid of radii
            density_grid : array_like
                The corresponding grid
        """
        MatterHalo.__init__(self)
        self.r_grid = r_grid
        self.density_grid = density_grid

    def density(self, r):
        """
        The (cubically) interpolated density function of the halo.
        The interpolation object is created each time to allow for changes in the density grid

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        rho = interp1d(self.r_grid, self.density_grid, kind='cubic', bounds_error=False, fill_value=(0.,0.))
        return np.where(r > self.r_min, rho(r), 0.)

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return "InterpolatedHalo"


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
        return "NFW"

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
        return "Spike"

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
        return "SpikedNFW"

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
        return "Hernquist"

class MichelAccretion(MatterHalo):
    """
    A class describing the baryonic accretion profile
        by the Michel solution, see section 11.4 of Rezzolla (2013)
        with an ideal gas polytropic EoS
            P = kappa * rho**Gamma

    Attributes:
        r_min (float): An minimum radius below which the density is always 0

        # These are the model parameters
        M     (float) : The mass of the central black hole
        kappa (float) : The polytropic constant
        Gamma (float) : The polytropic power law index

        # These values are needed for the ode initial condition
        r_c     (float) : The radius of the critical/sonic point
        rho_c   (float) : The density at the critical/sonic point
        u_c     (float) : The fluid velocity at the critical/sonic point

        # Other values for reference
        M_dot (float) : The mass accretion rate
    """

    def __init__(self, M, r_c, kappa, Gamma):
        """
        The constructor for the MichelAccretion class.
        Takes in the parameters and calculates the initial conditions needed for the ode integration

        Parameters:
            M : float
                The mass of the central black hole
            r_c : float
                The radius of the critical/sonic point
            kappa : float
                The constant of the polytropic EoS
            Gamma : float
                The power law index of the polytropic EoS
        """
        MatterHalo.__init__(self)

        self.M = M
        self.r_c = r_c
        self.kappa = kappa
        self.Gamma = Gamma

        # See equations in (11.94), (11.96), (11.101) in Rezzolla
        self.u_c = np.sqrt(M/2./r_c)
        c_s2 = self.u_c**2 / (1. - 3*self.u_c**2)
        self.rho_c = (1./Gamma/kappa * (Gamma-1.)*c_s2 / (Gamma-1.-c_s2))**(1./(Gamma-1.))
        self.M_dot = 4.*np.pi * self.r_c**2 * self.rho_c * self.u_c


    def FromM_dot(M, M_dot, rho_infty, kappa, Gamma):
        """
        An alternative constructor, using M_dot instead of theta_infty as an input
        This finds the corresponding theta_infty numerically and creates an object

        Parameters:
            M : float
                The mass of the central black hole
            M_dot : float
                The mass accretion rate
            rho_infty : float
                The density at infinity
            kappa : float
                The constant of the polytropic EoS
            Gamma : float
                The power law index of the polytropic EoS

        Returns:
            out : MichelAccretion
                The instantiated halo object

        TODO:
            Improve accuracy
        """
        # find solution for r_c
        lamb_c = 1./4. * ((5.-3.*Gamma)/2.)**((3.*Gamma-5.)/(2.*Gamma-2.))
        c_s_infty2 = (4.*np.pi*lamb_c*M**2 * rho_infty / M_dot)**(2./3.)

        def f(r_c):         # equation 11.98
            u_c2 = M/r_c/2.
            c_sc2 = u_c2/(1.-3.*u_c2)
            return ( (1.-c_sc2/(Gamma-1.))**2  /(1.-3.*u_c2)
                    - (1.- c_s_infty2 /(Gamma-1))**2 )

        #r= np.geomspace(1.*M, 1e5*M, 100)
        #plt.loglog(r/M, f(r)); plt.loglog(r/M, -f(r), linestyle='--'); plt.grid()

        bracket = [3.*M, 1e6*M]
        #print(f"f(1) = {f(bracket[0])}, f(2) = {f(bracket[1])}")
        sol = root_scalar(f, bracket=bracket, rtol=1e-7)
        r_c = sol.root
        #plt.axvline(r_c/M)
        return MichelAccretion(M, r_c, kappa, Gamma)


    def solve_ode(self, r):
        """
        Solves the differential equations for rho, u of the Michel system of equations
            at the requested radii, see
            https://arxiv.org/pdf/2102.12529.pdf
            for reference

        Parameters:
            r : float or array_like
                The radii at which rho, u are to be calculated, has to be strictly monotically increasing

        Returns:
            out : np.ndarray
                This is a multidimensional array where rho = out[0,:], u = out[1,:]  arrays correspond to r
        """
        rho_scale = self.rho_c
        u_scale = self.u_c
        def dy_dr(r, y):
            rho, u = y
            rho *= rho_scale; u *= u_scale;
            h = 1. + self.Gamma/(self.Gamma-1)*self.kappa * rho**(self.Gamma-1.)

            x = 1. - 2*self.M/r + u**2
            alpha = self.Gamma * self.kappa * rho**(self.Gamma-1.) * x / h / u**2
            du_dr = (2. * alpha * u**2 *r - self.M) / (u*r**2 * (1.-alpha))
            drho_dr = - rho/r/u * (r*du_dr + 2*u)
            #print(drho_dr, du_dr)
            return np.array([drho_dr/rho_scale, du_dr/u_scale])


        y_s = np.array([self.rho_c/rho_scale, self.u_c/u_scale])

        if r[0] >= self.r_c:
            sol = solve_ivp(dy_dr, [self.r_c, r[-1]], y_s, t_eval=r, atol=1e-13, rtol=1e-4)
            #print("1", sol.message)
            rho = sol.y[0,:]*rho_scale
            u = sol.y[1,:]*u_scale
            return rho, u
        elif r[-1] <= self.r_c:
            sol = solve_ivp(dy_dr , [self.r_c, r[0]], y_s, t_eval=r[::-1], atol=1e-13, rtol=1e-4)
            #print("2", sol.message)
            rho = (sol.y[0,:]*rho_scale)[::-1]
            u = (sol.y[1,:]*u_scale)[::-1]
            return rho,u
        else:
            rs= np.split(r, [np.where(r> self.r_c)[0][0]])
            rho1, u1 = self.solve_ode(rs[0])
            rho2, u2 = self.solve_ode(rs[1])
            return np.concatenate([rho1, rho2]), np.concatenate([u1,u2])

    # def solve_ode(self, r):
    #     """
    #     Solves the differential equations for rho, u of the Michel system of equations
    #         at the requested radii, see eq. (11.86) of Rezzolla (2013)

    #     Parameters:
    #         r : float or array_like
    #             The radii at which rho, u are to be calculated, has to be strictly monotically increasing

    #     Returns:
    #         out : np.ndarray
    #             This is a multidimensional array where rho = out[0,:], u = out[1,:]  arrays correspond to r
    #     """
    #     def du_dr(r, u):
    #         W2 = (1. - 2*self.M/r + u**2)
    #         rho = self.M_dot/(4.*np.pi*r**2 * u)
    #         cs2 = 1./ ( 1./(self.Gamma*self.kappa* rho**(self.Gamma-1.)) + 1./(self.Gamma-1.) )
    #         du_dr = - u/r * (2.*W2*cs2 - self.M/r) / (W2*cs2 - u**2)
    #         #print(f"r={r}, u={u}, W2={W2}, rho={rho}, cs2={cs2}, du_dr={du_dr}")
    #         return du_dr


    #     y_c = [self.u_c]

    #     if r[0] > self.r_c:
    #         sol = solve_ivp(du_dr, [self.r_c, r[-1]], y_c, t_eval=r, atol=1e-13, rtol=1e-5)
    #         print("1", sol.message)
    #         u = sol.y[0]
    #         rho = self.M_dot / (4.*np.pi*r**2 * u)
    #         return rho, u
    #     elif r[-1] < self.r_c:
    #         #sol = solve_ivp(lambda r,x: du_dr(r, np.exp(x))*np.exp(x) , [self.r_c, r[0]], np.log(y_c), t_eval=r[::-1], rtol=1e-6)
    #         sol = solve_ivp(lambda r,u: du_dr(r, u) , [self.r_c, r[0]], y_c, t_eval=r[::-1], atol=1e-13, rtol=1e-5)
    #         print("2", sol.message)
    #         u = sol.y[0][::-1]
    #         # sol = odeint(du_dr , y_c, np.append([self.r_c], r[::-1]), tfirst=True)
    #         # u = sol[1:,0][::-1]
    #         # print(u)
    #         rho = self.M_dot / (4.*np.pi*r**2 * u)
    #         return rho,u
    #     else:
    #         rs= np.split(r, [np.where(r> self.r_c)[0][0]])
    #         rho1, u1 = self.solve_ode(rs[0])
    #         rho2, u2 = self.solve_ode(rs[1])
    #         return np.concatenate([rho1, rho2]), np.concatenate([u1,u2])

    def machNumber(self, r):
        """
        Solves the differential equations for rho, u of the Michel system of equations
            and calculates the Mach number of the ideal gas

        Parameters:
            r : float or array_like
                The radii at which the mach number is to be calculated, has to be strictly monotically increasing

        Returns:
            out : np.ndarray
                The mach number at the point of interest
        """
        y = self.solve_ode(r)
        rho = y[0,:]
        u = y[1,:]
        V = u / np.sqrt(u**2 + 1.-2.*self.M/r )
        h = self.h_infty / np.sqrt(1. - 2.*self.M/r + u**2)
        C = np.sqrt(self.Gamma*self.kappa * rho**(self.Gamma-1) /h)
        return V*np.sqrt(1.-C**2) / C / np.sqrt(1. - V**2)


    def density(self, r):
        """
        The density function of the Michel accretion halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        if not  isinstance(r, (collections.Sequence, np.ndarray)):
            r = np.array([r])
        return self.solve_ode(r)[0]


    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return f"MichelAccretion(M={self.M}, M_dot={self.M_dot}, M_dot/M_dot_Edd={self.M_dot/(2.2 * 1e-9 * self.M /0.3064)}, rho_c={self.rho_c}, u_c={self.u_c}, kappa={self.kappa}, Gamma={self.Gamma})"



class DynamicSS(MatterHalo):
    """
    A class describing a spherically symmetric dynamic halo profile given a distribution function f inside a potential Phi
    The density is given by
        rho(r) = 4 \pi \int_0^v_max   v**2 * f(Phi(r) - v**2 /2)  dv


    Attributes:
        r_min (float): An minimum radius below which the density is always 0
        Eps_grid (array_like): The grid of relativ energy on which the distribution function f is given
        f_grid   (array_like): The corresponding values of the distribution function f
        potential(callable(r)): A reference to a callable function of r that gives the potential Phi
    """

    def __init__(self, Eps_grid, f_grid, potential):
        """
        The constructor for the DynamicSS class

        Parameters:
            Eps_grid : array_like
                The grid of relativ energy on which the distribution function f is given
            f_grid : array_like
                The corresponding values of the distribution function f
            potential : callable(r)
                A reference to a callable function of r that gives the potential Phi
        """
        MatterHalo.__init__(self)
        self.Eps_grid = Eps_grid
        self.f_grid = f_grid
        self.potential = potential
        self.update_Eps()

    def f(self, Eps):
        """
        This function returns linearly interpolated value(s) of the distribution function based on f_grid on the Eps_grid

        Parameters:
            Eps : float or array_like

        Returns:
            out : float or array_like
                The interpolated value(s) of the distribution function
        """
        return griddata(self.Eps_grid, self.f_grid, Eps, method='linear', fill_value=0.)

    '''
    # This is a previous idea to precompute the density on a grid when the distribution function is updated.
    #  This would allow many quick calls to the density profile afterwards. For the previous purposes
    #  of the HaloFeedback inspiral evolution, only one call to the density function is required, so this is mostly a waste of time.
    #  This could be useful when computing the mass profile tho
    def update_f(self):
        """
        This function updates an interpolation object of f and computes the density of the profile on a grid in radius that can be used to interpolate it later and possibly speed up computation

        Parameters:

        Returns:

        """
        #self.f_grid = np.clip(self.f_grid, 0., None)
        self._f = interp1d(self.Eps_grid, self.f_grid, kind='cubic', bounds_error=False, fill_value=(0., 0.))

        #ir_grid = self.r_of_Eps(self.Eps_grid)
        #v_max_grid = np.sqrt(2*self.potential(r_grid))
        #density = 4.*np.pi*np.array([quad(lambda v: v**2 * self.f(self.potential(r) - v**2 /2.), 0., v_max, limit=100)[0] for r, v_max in zip(r_grid, v_max_grid)])
        #self._density = interp1d(r_grid, density, kind='cubic', bounds_error=True)
    '''

    def update_Eps(self):
        """
        This function calculates the state density for the grid in relative energy and saves it in an interpolation object
        The state density is given by
            g(Eps) = 16 \pi^2 \int_0^r_{Eps} dr r^2 \sqrt{2 \Phi(r) - Eps}
        with the radius r_{Eps} = \Phi^{-1}(Eps)

        Parameters:

        Returns:

        TODO:
            Use a different r_grid that doesnt hardcode values
        """
        left, right = DynamicSS.FindEncompassingRBounds(self.Eps_grid, self.potential)
        r_grid = np.geomspace(left, right, int(10*np.log10(right/left)))
        self.r_of_Eps = interp1d(self.potential(r_grid), r_grid, kind='cubic', bounds_error=True)

        Eps_grid = self.Eps_grid
        stateDensity = 16.*np.pi**2 * np.array([quad(lambda r: r**2 * np.sqrt(np.clip(2.*self.potential(r) - 2.*Eps, 0., None)), 0., self.r_of_Eps(Eps), limit=100)[0] for Eps in Eps_grid])
        self._stateDensity = interp1d(Eps_grid, stateDensity, kind='cubic', bounds_error=False, fill_value=(np.inf, np.inf)) # Since we are dividing by the state density in the HaloModel, return np.inf for out of bounds


    def stateDensity(self, Eps):
        """
        This function returns the state density for a given Eps, based on an interpolation object
        The state density is given by
            g(Eps) = 16 \pi^2 \int_0^r_{Eps} dr r^2 \sqrt{2 \Phi(r) - Eps}
        with the radius r_{Eps} = \Phi^{-1}(Eps)

        Parameters:
            Eps : float or array_like
                The relative energy(s) of interest

        Returns:
            out : float or array_like
                The corresponding state density(s)
        """
        return self._stateDensity(Eps)
        '''
        # Alternative calculaltion without interpolation
        if not isinstance(Eps, (collections.Sequence, np.ndarray)):
            return 16.*np.pi**2 * quad(lambda r: r**2 * np.sqrt(2.*self.potential(r) - 2.*Eps), 0., self.r_of_Eps(Eps))[0]
        return 16.*np.pi**2 *np.array([quad(lambda r: r**2 * np.sqrt(2.*self.potential(r) - 2.*Eps), 0., self.r_of_Eps(Eps))[0]  for Eps in Eps])
        '''

    def density(self, r, v_max = None):
        """
        The density function of the dynamic halo depending on the distribution function
        The density is given by
            rho(r) = 4 \pi \int_0^v_max   v**2 * f(Phi(r) - v**2 /2)  dv

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density
            v_max : optional, float or array_like
                The maximum velocity up to which particles will be considered. Either a single value or array_like with the same size as r.
                The default value is v_max = \sqrt{2\Phi(r)} for a given radius.

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r

        TODO:
            Make design choice of integration method and test them
        """

        if v_max is None:
            v_max = np.sqrt(2*self.potential(r))
        if not isinstance(r, (collections.Sequence, np.ndarray)):
            v2_list = np.linspace(0., v_max**2, 3000)
            f_list = self.f(self.potential(r) - 0.5*v2_list)
            return 4.*np.pi*simps(v2_list * f_list, x=np.sqrt(v2_list))
            #return 4.*np.pi*quad(lambda v: v**2 * self.f(self.potential(r) - v**2 /2.), 0., v_max, limit=200)[0]

        if not isinstance(v_max, (collections.Sequence, np.ndarray)):
            v_max = [v_max]*len(r)
        v_max = np.clip(v_max, 0., np.sqrt(2*self.potential(r)))
        return np.array([self.density(r, v) for r,v in zip(r, v_max) ])
        #return 4.*np.pi*np.array([quad(lambda v: v**2 * self.f(self.potential(r[i]) - v**2 /2.), 0., v_max[i], limit=200)[0] for i in range(len(r))])

    def FindEncompassingRBounds(Eps_grid, potential):
        left = -5.; right=5.
        while potential(10**left) < Eps_grid[-1] and left > -20:
            left -= 1
        while potential(10.**right) > Eps_grid[0] and right < 30:
            right += 1
        return (10**left, 10**right)


    def FromStatic(Eps_grid, halo, extPotential):
        """
        This function calculates the distribution function f, given a grid in relative energy, a density profil and the potential in which this is located.
        It implements the Eddington inversion described by [Jo Bovy](http://astro.utoronto.ca/~bovy/AST1420/notes/notebooks/05.-Equilibria-Spherical-Collisionless-Systems.html#Ergodic-DFs-and-the-Eddington-inversion-formula)

        The distribution function is given by
            f(Eps) = 1/ \sqrt(8)/ \pi**2 \int_0^{Eps} 1/ \sqrt(Eps - Phi)  d^2 \\rho/ dPhi^2

        Parameters:
            Eps_grid : array_like
                The grid in relative energy on which to calculate the distribution function
            halo : MatterHalo object
                The MatterHalo object that implements the density profile that is to be inverted
            extPotential : callable(r)
                The potential in which the density profile is to be inverted

        Returns:
            out : DynamicSS object
                The DynamicSS object with the corresponding distribution function on the grid in relative energy

        """
        # find the right r grid for calculations, such that the potential encompasses the Eps_grid
        left, right = DynamicSS.FindEncompassingRBounds(Eps_grid, extPotential)
        r = np.geomspace(left, right, int(10*np.log10(right/left)))
        # calculate f according to Eddington inversion
        Phi = extPotential(r)

        ''' # First possible method
        drho_dphi = interp1d(Phi, np.gradient(halo.density(r),  Phi), kind='cubic', fill_value=(0.,0.), bounds_error=False)
        intdPhi = np.array([quad(lambda p: drho_dphi(p)/np.sqrt(Eps-p), 0., Eps, limit=200)[0] for Eps in Eps_grid])
        f_grid = 1./np.sqrt(8.)/np.pi**2 * np.gradient(intdPhi, Eps_grid)
        '''
         # Second possible method
        drho_dphi =  np.gradient(halo.density(r),  Phi)
        d2rho_dphi2 = interp1d(Phi, np.gradient(drho_dphi, Phi), kind='cubic', fill_value=(0.,0.), bounds_error=False)
        f_grid = 1./np.sqrt(8.)/np.pi**2 * np.array([ quad(lambda p: d2rho_dphi2(p) / np.sqrt(E - p), 0., E, limit=200)[0] for E in Eps_grid])

        # The Eddington inversion is not guaranteed to give a physical solution
        f_grid[np.where(f_grid < 0.)] =  0.
        return DynamicSS(Eps_grid, f_grid, extPotential)

    def FromSpike(Eps_grid, sp, spike):
        """
        This function implements the analytically known distribution function f of a power-law spike, given a grid in relative energy.
        The analytic equation is only valid in a central potential of m1, which is extracted from the SystemProp object.

        Parameters:
            Eps_grid : array_like
                The grid in relative energy on which to calculate the distribution function
            sp  : merger_system.SystemProp
                The object that contains info about the binary system, m1 is extracted from this
            spike : Spike object
                The Spike object that contains the spike parameters

        Returns:
            out : DynamicSS object
                The DynamicSS object with the corresponding distribution function on the grid in relative energy

        """
        extPotential = lambda r: sp.m1/r
        f_grid =  (spike.rho_spike * spike.alpha*(spike.alpha-1.)/(2.*np.pi)**(3./2.)
            * (spike.r_spike/sp.m1)**spike.alpha * gamma(spike.alpha-1.)/gamma(spike.alpha-1./2.)
            * Eps_grid**(spike.alpha-3./2.) )
        return DynamicSS(Eps_grid, f_grid, extPotential)

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return "DynamicSS"

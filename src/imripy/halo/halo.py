import numpy as np
from scipy.interpolate import interp1d, griddata
from scipy.integrate import odeint, quad, simps, solve_ivp
from scipy.special import gamma
#import matplotlib.pyplot as plt
from collections.abc import Sequence
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
        if isinstance(r, (Sequence, np.ndarray)):
            return 4.*np.pi*odeint(integrand, quad(integrand, 0., r[0], args=(0.))[0], r, tfirst=True, rtol=1e-10, atol=1e-10)[:,0]
        else:
            return 4.*np.pi*quad(integrand, 0., r, args=(0.))[0]

    def velocity(self, r):
        """
        The velocity of the particles in the halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the velocity

        Returns:
            out : float or array_like (depending on r)
                The velocity at the radius r
        """
        return 0.

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return "MatterHalo"



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
        name   (string)     : The name of the halo being interpolated
    """

    def __init__(self, r_grid, density_grid, name=""):
        """
        The constructor for the ConstHalo class

        Parameters:
            r_grid : array_like
                The grid of radii
            density_grid : array_like
                The corresponding grid
            name : string  (optional)
                The name of the halo being interpolated
        """
        MatterHalo.__init__(self)
        self.r_grid = r_grid
        self.density_grid = density_grid
        self.name = name

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
        return "InterpolatedHalo" + ( (" (" + self.name + ")") if len(self.name) > 0 else "")



class MatterHaloDF(MatterHalo):
    """
    A placeholder class describing a spherically symmetric halo profile given a distribution function f inside a potential Phi
    The density is given by
        rho(r) = 4 \pi \int_0^v_max   v**2 * f(Phi(r) - v**2 /2)  dv
    """

    def __init__(self):
        """
        The constructor for the MatterHaloDF class

        """
        MatterHalo.__init__(self)

    def f(self, Eps):
        """
        This function is a placeholder for the distribution function

        Parameters:
            Eps : float or array_like

        Returns:
            out : float or array_like
                The value(s) of the distribution function at the given energy densities
        """
        pass

    def potential(self, r):
        """
        This is a placeholder function for the potential

        Parameters:
            r : float or array_like

        Returns:
            out : float or array_like
                The value(s) of the potential at the given radius
        """
        pass


    def stateDensity(self, Eps):
        """
        This function returns the state density for a given Eps
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
        pass
        '''
        # Alternative calculaltion without interpolation
        if not isinstance(Eps, (Sequence, np.ndarray)):
            return 16.*np.pi**2 * quad(lambda r: r**2 * np.sqrt(2.*self.potential(r) - 2.*Eps), 0., self.r_of_Eps(Eps))[0]
        return 16.*np.pi**2 *np.array([quad(lambda r: r**2 * np.sqrt(2.*self.potential(r) - 2.*Eps), 0., self.r_of_Eps(Eps))[0]  for Eps in Eps])
        '''

    def density(self, r, v_max = None):
        """
        The density function of the halo depending on the distribution function
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
        if not isinstance(r, (Sequence, np.ndarray)):
            v2_list = np.linspace(0., v_max**2, 3000)
            f_list = self.f(self.potential(r) - 0.5*v2_list)
            return 4.*np.pi*simps(v2_list * f_list, x=np.sqrt(v2_list))
            #return 4.*np.pi*quad(lambda v: v**2 * self.f(self.potential(r) - v**2 /2.), 0., v_max, limit=200)[0]

        if not isinstance(v_max, (Sequence, np.ndarray)):
            v_max = np.ones(len(r))*v_max
        v_max = np.clip(v_max, 0., np.sqrt(2*self.potential(r)))
        return np.array([self.density(r, v) for r,v in zip(r, v_max) ])
        #return 4.*np.pi*np.array([quad(lambda v: v**2 * self.f(self.potential(r[i]) - v**2 /2.), 0., v_max[i], limit=200)[0] for i in range(len(r))])


    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return "MatterHaloDF"


class DynamicSS(MatterHaloDF):
    """
    A class describing a spherically symmetric dynamic halo profile given a distribution function f inside a potential Phi
    Here, the function f is interpolated from a grid in f_grid, Eps_grid
    The density is given by
        rho(r) = 4 \pi \int_0^v_max   v**2 * f(Phi(r) - v**2 /2)  dv


    Attributes:
        r_min (float): An minimum radius below which the density is always 0
        Eps_grid (array_like): The grid of relativ energy on which the distribution function f is given
        f_grid   (array_like): The corresponding values of the distribution function f
        potential(callable(r)): A reference to a callable function of r that gives the potential Phi
        interpolate_density: Whether to interpolate the density once and call that instead of recomputing  # TODO
    """

    def __init__(self, Eps_grid, f_grid, potential, interpolate_density=False):
        """
        The constructor for the DynamicSS class

        Parameters:
            Eps_grid : array_like
                The grid of relativ energy on which the distribution function f is given
            f_grid : array_like
                The corresponding values of the distribution function f
            potential : callable(r)
                A reference to a callable function of r that gives the potential Phi
            interpolate_density : Boolean
                Whether to precompute and later interpolate the density function instead of doing the integral over f each time
        """
        MatterHaloDF.__init__(self)
        self.Eps_grid = Eps_grid
        self.f_grid = f_grid
        self.potential = potential
        self.interpolate_density = interpolate_density
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
        if not isinstance(Eps, (Sequence, np.ndarray)):
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
        """
        if self.interpolate_density:
            pass # TODO
        else:
            return super().density(r, v_max)

    def FindEncompassingRBounds(Eps_grid, potential):
        left = -5.; right=5.
        while potential(10**left) < Eps_grid[-1] and left > -20:
            left -= 1
        while potential(10.**right) > Eps_grid[0] and right < 30:
            right += 1
        return (10**left, 10**right)


    def EddingtonInversion(Eps_grid, halo, extPotential):
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

    def FromStatic(Eps_grid, halo, extPotential=None):
        """"
        This function takes a MatterHalo or MatterHaloDF object and makes a grid in f over the given Eps_grid
        If it is a MatterHaloDF object, f is taken from the analytic equations.
        If it is a MatterHalo object, f is calculated through the Eddingtion inversion procedure, see the EddingtonInversion function

        Parameters:
            Eps_grid : array_like
                The grid in relative energy on which to calculate the distribution function
            halo : MatterHalo(DF) object
                The MatterHalo object from which to extract the distribution function
            extPotential : (optional) callable(r)
                The potential in which the density profile is to be inverted - ignored if provided by MatterHaloDF object

        Returns:
            out : DynamicSS object
                The DynamicSS object with the corresponding distribution function on the grid in relative energy
        """
        if hasattr(halo, "f"):
            return DynamicSS(Eps_grid, halo.f(Eps_grid), halo.potential if hasattr(halo, "potential") else extPotential)
        else:
            return DynamicSS.EddingtonInversion(Eps_grid, halo, extPotential=extPotential)

    def FromSpike(Eps_grid, sp, spike):
        """
        Deprecated, use FromStatic method
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
        spike.M_bh = sp.m1
        return DynamicSS.FromStatic(Eps_grid, spike)

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return "DynamicSS"

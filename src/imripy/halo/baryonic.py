from .halo import *
import imripy.constants as c

from scipy.optimize import root_scalar, root


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

        # See equations in (11.94), (11.96), (11.97), (11.101) in Rezzolla (2013)
        self.u_c = np.sqrt(M/2./r_c)
        c_s2 = self.u_c**2 / (1. - 3*self.u_c**2)
        self.rho_c = (1./Gamma/kappa * (Gamma-1.)*c_s2 / (Gamma-1.-c_s2))**(1./(Gamma-1.))
        self.M_dot = 4.*np.pi * self.r_c**2 * self.rho_c * self.u_c
        self.rho_infty = (((1. + Gamma*kappa/(Gamma-1.)*self.rho_c**(Gamma-1.))*np.sqrt(1.-2.*self.M/self.r_c + self.u_c**2)
              -1.) * (Gamma-1.)/Gamma/kappa)**(1./(Gamma-1.))


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


    # def solve_ode(self, r):
    #     """
    #     Solves the differential equations for rho, u of the Michel system of equations
    #         at the requested radii, see
    #         https://arxiv.org/pdf/2102.12529.pdf
    #         for reference

    #     Parameters:
    #         r : float or array_like
    #             The radii at which rho, u are to be calculated, has to be strictly monotically increasing

    #     Returns:
    #         out : np.ndarray
    #             This is a multidimensional array where rho = out[0,:], u = out[1,:]  arrays correspond to r
    #     """
    #     rho_scale = self.rho_c
    #     u_scale = self.u_c
    #     def dy_dr(r, y):
    #         rho, u = y
    #         rho *= rho_scale; u *= u_scale;
    #         h = 1. + self.Gamma/(self.Gamma-1)*self.kappa * rho**(self.Gamma-1.)

    #         x = 1. - 2*self.M/r + u**2
    #         alpha = self.Gamma * self.kappa * rho**(self.Gamma-1.) * x / h / u**2
    #         du_dr = (2. * alpha * u**2 *r - self.M) / (u*r**2 * (1.-alpha))
    #         drho_dr = - rho/r/u * (r*du_dr + 2*u)
    #         #print(drho_dr, du_dr)
    #         return np.array([drho_dr/rho_scale, du_dr/u_scale])


    #     y_s = np.array([self.rho_c/rho_scale, self.u_c/u_scale])

    #     if r[0] >= self.r_c:
    #         sol = solve_ivp(dy_dr, [self.r_c, r[-1]], y_s, t_eval=r, atol=1e-13, rtol=1e-4)
    #         #print("1", sol.message)
    #         rho = sol.y[0,:]*rho_scale
    #         u = sol.y[1,:]*u_scale
    #         return rho, u
    #     elif r[-1] <= self.r_c:
    #         sol = solve_ivp(dy_dr , [self.r_c, r[0]], y_s, t_eval=r[::-1], atol=1e-13, rtol=1e-4)
    #         #print("2", sol.message)
    #         rho = (sol.y[0,:]*rho_scale)[::-1]
    #         u = (sol.y[1,:]*u_scale)[::-1]
    #         return rho,u
    #     else:
    #         rs= np.split(r, [np.where(r> self.r_c)[0][0]])
    #         rho1, u1 = self.solve_ode(rs[0])
    #         rho2, u2 = self.solve_ode(rs[1])
    #         return np.concatenate([rho1, rho2]), np.concatenate([u1,u2])

    def solve_ode(self, r):
        """
        Solves the differential equations for rho, u of the Michel system of equations
            at the requested radii, see eq. (11.86) of Rezzolla (2013)

        Parameters:
            r : float or array_like
                The radii at which rho, u are to be calculated, has to be strictly monotically increasing

        Returns:
            out : np.ndarray
                This is a multidimensional array where rho = out[0,:], u = out[1,:]  arrays correspond to r
        """
        def du_dr(r, u):
            W2 = (1. - 2*self.M/r + u**2)
            rho = self.M_dot/(4.*np.pi*r**2 * u)
            cs2 = 1./ ( 1./(self.Gamma*self.kappa* rho**(self.Gamma-1.)) + 1./(self.Gamma-1.) )
            du_dr = - u/r * (2.*W2*cs2 - self.M/r) / (W2*cs2 - u**2)
            #print(f"r={r}, u={u}, W2={W2}, rho={rho}, cs2={cs2}, du_dr={du_dr}")
            return du_dr


        y_c = [self.u_c]

        if r[0] > self.r_c:
            sol = solve_ivp(du_dr, [self.r_c, r[-1]], y_c, t_eval=r, atol=1e-13, rtol=1e-5)
            #print("1", sol.message)
            u = sol.y[0]
            rho = self.M_dot / (4.*np.pi*r**2 * u)
            return rho, u
        elif r[-1] < self.r_c:
            #sol = solve_ivp(lambda r,x: du_dr(r, np.exp(x))*np.exp(x) , [self.r_c, r[0]], np.log(y_c), t_eval=r[::-1], rtol=1e-6)
            sol = solve_ivp(lambda r,u: du_dr(r, u) , [self.r_c, r[0]], y_c, t_eval=r[::-1], atol=1e-13, rtol=1e-5)
            #print("2", sol.message)
            u = sol.y[0][::-1]
            # sol = odeint(du_dr , y_c, np.append([self.r_c], r[::-1]), tfirst=True)
            # u = sol[1:,0][::-1]
            # print(u)
            rho = self.M_dot / (4.*np.pi*r**2 * u)
            return rho,u
        else:
            rs= np.split(r, [np.where(r> self.r_c)[0][0]])
            rho1, u1 = self.solve_ode(rs[0])
            rho2, u2 = self.solve_ode(rs[1])
            return np.concatenate([rho1, rho2]), np.concatenate([u1,u2])

    def velocity(self, r):
        """
        The velocities v_r, v_phi of the particles in the halo

        The density function of the Michel accretion halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            v_r : array_like
                The radial velocity at the radius r
            v_phi : array_like
                The angular velocity at the radius r
        """
        if not  isinstance(r, (Sequence, np.ndarray)):
            r = np.array([r])
        u = self.solve_ode(r)[1]
        W = np.sqrt(1. - 2.*self.M/r + u**2)
        v_r = -u/W  # see eq (11.98) in Rezzolla(2013) but with a negative sign since the particles are falling inward
        v_r = v_r[0] if len(v_r) == 1 else v_r
        v_phi = np.zeros(np.shape(v_r))
        return (v_r, v_phi)

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
        if not  isinstance(r, (Sequence, np.ndarray)):
            r = np.array([r])
        return self.solve_ode(r)[0]

    def soundspeed(self, r):
        """
        The soundspeed c_s of the Michel accretion halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the soundspeed

        Returns:
            out : float or array_like (depending on r)
                The soundspeed at the radius r
        """
        return (1./ self.Gamma/self.kappa/self.density(r)**(self.Gamma-1.) + 1./(self.Gamma-1.))**(-1./2.)

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return f"MichelAccretion(M={self.M}, M_dot={self.M_dot}, M_dot/M_dot_Edd={self.M_dot/(2.2 * 1e-9 * self.M /0.3064)}, rho_c={self.rho_c}, u_c={self.u_c}, kappa={self.kappa}, Gamma={self.Gamma})"


class BaryonicDisk(MatterHalo):
    """
    An abstract class designed to streamline implementation of different baryonic disk models. Inherits from MatterHalo, but is axially symmetric, not spherically symmetric.
    Has additional functions usually used to describe baryonic disk

    """
    def __init__(self):
        super().__init__()

    def density(self, r, z=0.):
        pass

    def surface_density(self, r):
        pass

    def scale_height(self, r):
        pass

    def mach_number(self, r):
        pass

    def soundspeed(self, r):
        pass

    def velocity(self, r, phi, z=0.):
        pass

    def mass(self, r, **kwargs):
        """
        The mass that is contained in the disk in a circular shell of size r.
        This function numerically integrates over the density function, so that it can be used
            in inherited classes that have not implemented an analytic expression for the mass.

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or np.ndarray (depending on r)
                The mass inside the disk of size r
        """
        integrand = lambda r, m: self.surface_density(r, **kwargs)*r
        if isinstance(r, (Sequence, np.ndarray)):
            return 2.*np.pi*odeint(integrand, quad(integrand, 0., r[0], args=(0.))[0], r, tfirst=True, rtol=1e-10, atol=1e-10)[:,0]
        else:
            return 2.*np.pi*quad(integrand, 0., r, args=(0.))[0]

    def Q(self, r):
        """
        Toomre's stability parameter Q for the disk
        """
        Omega = np.sqrt(self.M/r**3)
        #return self.soundspeed(r) * Omega / np.pi / self.surface_density(r)
        return  Omega**2 / 2. / np.pi / self.density(r, z=0.)

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return "BaryonicDisk"


class AlphaDisk(BaryonicDisk):
    """
    The class describing a baryonic accretion disk as introduced by Shakura & Sunyaev
        with the alpha disk parametrization as explained in https://arxiv.org/pdf/2207.10086.pdf

    Attributes:
        r_min (float): An minimum radius below which the density is always 0

        # These are the model parameters
        M     (float) : The mass of the central black hole
        alpha (float) : The viscosity parameter
        f_edd (flaot) : The Eddington accretion rate fraction
        eps   (float) : mass-energy conversion efficiency
            (technically we don't need both of these parameters, this is just for convenience)
        r_max (float) : the maximal radius of the disk
    """

    def __init__(self, M, alpha, f_edd, eps, r_max = np.inf):
        """
        The constructor for the AlphaDisk class.

        Parameters:
            M : float
                The mass of the central black hole
            alpha : float
                The viscosity coeffcient
            f_edd : float
                The Eddintion accretion rate fraction
            eps   : float
                The mass-energy conversion efficiency
            r_max : float
                The maximal radius of the disk
        """
        super().__init__()
        self.M = M
        self.alpha = alpha
        self.f_edd = f_edd
        self.eps = eps
        self.r_max = r_max

    def surface_density(self, r):
        """
        The surface density function of the disk
            as given by eq (2.1) of https://arxiv.org/pdf/2207.10086.pdf

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The surface density at the radius r
        """
        # convert kg/m^2 to invpc
        return np.where( r < self.r_max,
                            5.4e3 * 0.1*c.g_cm2_to_invpc / (self.alpha/0.1) / (self.f_edd / 0.1 * 0.1 / self.eps) * (r/self.M)**(3./2.),
                             0.)

    def scale_height(self, r):
        """
        The disk scale height at radius r
            as given by eq (2.1) of https://arxiv.org/pdf/2207.10086.pdf

        Parameters:
            r : float or array_like
                The radius at which to evaluate the scale height

        Returns:
            out : float or array_like (depending on r)
                The disk scale height at the radius r
        """
        return np.where( r < self.r_max,
                            1.5 * (self.f_edd/0.1 * 0.1 / self.eps) * self.M * np.ones(np.shape(r)),
                            0.)

    def density(self, r, z=0.):
        """
        The density function of the disk

        Parameters:
            r : float
                The radius inside the disk plane
            z : float
                The height above the disk

        Returns:
            out : float
                The density at the radius r
        """
        return np.where( r < self.r_max,
                            self.surface_density(r) / 2. / self.scale_height(r) * np.exp(-z**2 / self.scale_height(r)**2),
                            0.)

    def soundspeed(self, r):
        """
        The soundspeed c_s of the disk
            as given by eq (A6) of https://arxiv.org/pdf/2206.05292.pdf

        Parameters:
            r : float or array_like
                The radius at which to evaluate the soundspeed

        Returns:
            out : float or array_like (depending on r)
                The soundspeed at the radius r
        """
        Omega = np.sqrt(self.M/r**3)
        return np.where( r < self.r_max,
                            self.scale_height(r) * Omega,
                            0.)

    def velocity(self, r, phi, z=0.):
        """
        The velocity of the particles in the disk

        Parameters:
            r : float
                The cylindrical radius inside the disk
            phi : float
                The cylindrical angle inside the disk
            z   : float
                The height above the disk plane

        Returns:
            out : tuple
                The velocity [v_X, v_Y, v_Z] in the fundamental plane
        """
        v_phi = np.sqrt(self.M/r)
        return  np.array([-v_phi*np.sin(phi) , v_phi*np.cos(phi), 0.])


    def mach_number(self, r):
        """
        The mach number of the disk at radius r
            as given below eq (A7) of https://arxiv.org/pdf/2206.05292.pdf

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mach number

        Returns:
            out : float or array_like (depending on r)
                The mach number at the radius r
        """
        return np.where( r < self.r_max,
                            r / self.scale_height(r),
                            0.)

    def __str__(self):
        return f"AlphaDisk (M = {self.M}, alpha={self.alpha}, f_edd/eps={self.f_edd/self.eps})"


class BetaDisk(BaryonicDisk):
    """
    The class describing a baryonic accretion disk as introduced by Shakura & Sunyaev
        with the beta disk parametrization as explained in https://arxiv.org/pdf/2207.10086.pdf

    Attributes:
        r_min (float): An minimum radius below which the density is always 0

        # These are the model parameters
        M     (float) : The mass of the central black hole
        alpha (float) : The viscosity parameter
        f_edd (flaot) : The Eddington accretion rate fraction
        eps   (float) : mass-energy conversion efficiency
            (technically we don't need both of these parameters, this is just for convenience)
        r_max (float) : the maximal radius of the disk
    """

    def __init__(self, M, alpha, f_edd, eps, r_max=np.inf):
        """
        The constructor for the BetaDisk class.

        Parameters:
            M : float
                The mass of the central black hole
            alpha : float
                The viscosity coeffcient
            f_edd : float
                The Eddintion accretion rate fraction
            eps   : float
                The mass-energy conversion efficiency
            r_max : float
                The maximal radius of the disk
        """
        super().__init__()
        self.M = M
        self.alpha = alpha
        self.f_edd = f_edd
        self.eps = eps
        self.r_max = r_max

    def surface_density(self, r):
        """
        The surface density function of the disk
            as given by eq (2.1) of https://arxiv.org/pdf/2207.10086.pdf

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The surface density at the radius r
        """
        # convert kg/m^2 to invpc
        return  np.where( r < self.r_max,
                            (2.1e7 * 0.1*c.g_cm2_to_invpc * (self.alpha/0.1)**(-4./5.) * (self.f_edd/self.eps)**(3./5.)
                                * (self.M / 1e6 / c.solar_mass_to_pc)**(1./5.) * (r / 10./self.M)**(-3./5.)),
                            0.)

    def scale_height(self, r):
        """
        The disk scale height at radius r
            as given by eq (2.1) of https://arxiv.org/pdf/2207.10086.pdf

        Parameters:
            r : float or array_like
                The radius at which to evaluate the scale height

        Returns:
            out : float or array_like (depending on r)
                The disk scale height at the radius r
        """
        return np.where( r < self.r_max,
                            1.5 * (self.f_edd/0.1 * 0.1 / self.eps) * self.M* np.ones(np.shape(r)),
                            0.)

    def density(self, r, z=0.):
        """
        The density function of the disk

        Parameters:
            r : float
                The radius inside the disk plane
            z : float
                The height above the disk

        Returns:
            out : float
                The density at the radius r
        """
        return np.where( r < self.r_max,
                            self.surface_density(r) / 2. / self.scale_height(r) * np.exp(-z**2 / self.scale_height(r)**2),
                            0.)

    def soundspeed(self, r):
        """
        The soundspeed c_s of the disk
            as given by eq (A6) of https://arxiv.org/pdf/2206.05292.pdf

        Parameters:
            r : float or array_like
                The radius at which to evaluate the soundspeed

        Returns:
            out : float or array_like (depending on r)
                The soundspeed at the radius r
        """
        Omega = np.sqrt(self.M/r**3)
        return np.where(r < self.r_max,
                            self.scale_height(r) * Omega,
                            0.)

    def velocity(self, r, phi, z=0.):
        """
        The velocity of the particles in the disk

        Parameters:
            r : float
                The cylindrical radius inside the disk
            phi : float
                The cylindrical angle inside the disk
            z   : float
                The height above the disk plane

        Returns:
            out : tuple
                The velocity [v_X, v_Y, v_Z] in the fundamental plane
        """
        v_phi = np.sqrt(self.M/r)
        return  np.array([-v_phi*np.sin(phi) , v_phi*np.cos(phi), 0.])

    def mach_number(self, r):
        """
        The mach number of the disk at radius r
            as given below eq (A7) of https://arxiv.org/pdf/2206.05292.pdf

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mach number

        Returns:
            out : float or array_like (depending on r)
                The mach number at the radius r
        """
        return np.where( r < self.r_max,
                            r / self.scale_height(r),
                            0.)

    def __str__(self):
        return f"BetaDisk (M = {self.M}, alpha={self.alpha}, f_edd/eps={self.f_edd/self.eps})"


class DerdzinskiMayerDisk(BaryonicDisk):
    """
    The class describing a baryonic accretion disk as introduced by Derdzinski & Mayer,
        based on models of Shakura & Sunyaev and Sirko & Goodman, using an opacity description by Bell & Linn,
        with viscosity and opacity descriptions as given by the equations of appendix A of https://arxiv.org/pdf/2206.05292.pdf

    Attributes:
        r_min (float): An minimum radius below which the density is always 0

        # These are the model parameters
        M     (float) : The mass of the central black hole
        M_dot (float) : The accretion rate of the central black hole
        alpha (float) : The viscosity parameter <1

        # These constants are needed for computation
        boltzmann_constant  (float) : in units of pc/Kelvin
        stefan_boltzmann_constant   (float) : in 1/pc^2/Kelving^4
        hydrogen_mass   (float) : in pc
        a_rad   (float) : radiation density constant
    """

    boltzmann_constant = 3.697e-84  # in pc / kelvin
    stefan_boltzmann_constant = 1.563e-60 / c.m_to_pc**2  # in 1/pc^2 / Kelvin^4
    mean_molecular_weight = 0.62
    hydrogen_mass = 1.243e-54 * c.m_to_pc  # in pc


    def __init__(self, M, M_dot, alpha):
        """
        The constructor for the DerdzinskiMayerDisk class.

        Parameters:
            M : float
                The mass of the central black hole
            M_dot : float
                The accretion rate
            alpha : float
                The viscosity coefficient <1
        """
        BaryonicDisk.__init__(self)
        self.M = M
        self.M_dot = M_dot
        self.alpha = alpha
        self.r_min = 6.*M

    def opacity_scaling(self, rho, T):
        """
        Calculates the opacity scaling of an accretion disk depending on temperature and density
        Values taken from https://arxiv.org/pdf/2205.10382.pdf

        Parameters:
            rho : floa
                The density in units of TODO
            T   : float
                The temperature in units of Kelvin

        Returns:
            out : float
                The Rosseland mean opacity
        """
        kappa_0 = 0.; a = 0.; b = 0.
        if T < 166.81:
            kappa_0 = 2e-4;     a = 0; b = 2.
        elif T < 202.677:
            kappa_0 = 2e16;     a = 0; b = -7.
        elif T < 2286.77 * rho**(2./49.) :
            kappa_0 = 1e-1;     a = 0; b = 1./2.
        elif T < 2029.76 * rho**(1./81.) :
            kappa_0 = 2e81;     a = 1; b = -24.
        elif T < 1e4 * rho**(1./21.) :
            kappa_0 = 1e-8;     a = 2./3.; b = 3.
        elif T < 31195.2 * rho**(4./75.) :
            kappa_0 = 1e-36;    a = 1./3.; b = 10.
        elif T < 1.79393e8 * rho**(2./5.) :
            kappa_0 = 1.5e20;   a = 1; b = -5./2.
        elif T > 2e-4:
            kappa_0 = 0.348;    a = 0.; b = 0.
        #print(kappa_0, a, b, rho**a, T**b)
        return kappa_0 * rho**a * T**b



    def solve_eq(self, r, rho_0=None, Sigma_0=None, T_mid_0=None, c_s2_0=None):
        """
        Solves the nonlinear equations of density, surface density, temperature and sound speed describing the disk
            as given in appendix A of https://arxiv.org/pdf/2206.05292.pdf

        Parameters:
            r : float
                The radius of the point of interest
            *_0   (optional): float
                The initial guesses for the values of interest
                can e.g. be passed if a point nearby is known already

        Returns:
            rho: float
                The density at the radius r
            Sigma : float
                The surface density at radius r
            T_mid : float
                The midplane temperature
            c_s2   : float
                The soundspeed squared
        """
        if r <= self.r_min:
            return 0., 0., 0., 0.
        # calculate knowns at the given radius
        Omega = np.sqrt(self.M/r**3)
        M_dot_prime = self.M_dot * (1. - np.sqrt(self.r_min/r))
        T_eff =  (3./8./np.pi * Omega**2 * M_dot_prime / self.stefan_boltzmann_constant)**(1./4.)
        Q_min = 1.4
        rho_max = Omega**2 / 2. / np.pi / Q_min
        # print(f"Omega = {Omega:.3e}, M_dot_prime={M_dot_prime:.3e}, T_eff={T_eff:.3e}, rho_max={rho_max:.3e}")

        # solve for rho, Sigma, T_mid, and c_s^2
        def f(x):
            # use exponential function to assert positivity, and logistic function to assert 0<value<max
            sm_rho, ln_Sigma, ln_T_mid, sm_c_s2 = x
            rho = rho_max/(1.+np.exp(-sm_rho)); Sigma = np.exp(ln_Sigma); T_mid = np.exp(ln_T_mid); c_s2 = 1./(1.+np.exp(-sm_c_s2))

            Q = np.sqrt(c_s2) * Omega / np.pi / Sigma
            nu = self.alpha * c_s2 / Omega
            kappa = self.opacity_scaling(rho/c.g_cm3_to_invpc2, T_mid) / c.g_cm2_to_invpc # TODO: check units
            tau_opt = kappa*Sigma/2.
            h = np.sqrt(c_s2)/Omega
            p_gas = self.boltzmann_constant / self.hydrogen_mass / self.mean_molecular_weight * rho * T_mid
            p_rad = 1./2. * self.stefan_boltzmann_constant * tau_opt * T_eff**4

            # target values
            rho_t = Sigma / 2. / h if Q > 1.5*Q_min else rho_max*(1.-1e-10) # assert stability in outer parts
            Sigma_t = M_dot_prime / 3. / np.pi / nu
            c_s2_t = (p_gas + p_rad)/ rho
            T_mid_t = (3./8. * tau_opt + 1./2. + 1./4./tau_opt)**(1./4.) * T_eff
            # print(f"kappa = {kappa*c.g_cm2_to_invpc}, tau_opt = {tau_opt}")
            # print(f"rho={rho:.3e}->{rho_t:.3e}, Sigma={Sigma:.3e}->{Sigma_t:.3e}, T_mid={T_mid:.3e}->{T_mid_t:.3e}, c_s2={c_s2:.3e}->{c_s2_t:.3e}")
            return np.array([np.log(rho_t/(rho_max-rho_t)) - sm_rho, np.log(Sigma_t) - ln_Sigma, np.log(T_mid_t) - ln_T_mid,  np.log(c_s2_t/(1.-c_s2_t)) - sm_c_s2])

        # choose initial values
        mach_number_0 = 60.
        Sigma_0 = 1e3 * c.g_cm2_to_invpc if Sigma_0 is None else Sigma_0
        rho_0 = Sigma_0 / 2. / r * mach_number_0 if rho_0 is None else rho_0
        rho_0 = np.clip(rho_0, 1e-20, rho_max*(1.-1e-10))
        T_mid_0 = T_eff if T_mid_0 is None else T_mid_0
        c_s2_0 = (self.boltzmann_constant / self.hydrogen_mass / self.mean_molecular_weight * T_mid_0
                             +  4./3. * self.stefan_boltzmann_constant* T_mid_0**4 / rho_0 )  if c_s2_0 is None else c_s2_0

        # compute solution
        x_0 = np.array([ np.log(rho_0/(rho_max-rho_0)), np.log(Sigma_0), np.log(T_mid_0), np.log(c_s2_0/(1.-c_s2_0))])
        sol = root(f, x0 = x_0, method='lm', options={'xtol':1e-10,'ftol':1e-10})
        if not sol.success:
            print(sol.message, r, x_0, sol.x)

        rho, Sigma, T_mid, c_s2 = np.exp(sol.x)
        rho = rho_max /(1. + 1./rho)
        c_s2 = 1./(1.+1./c_s2)

        return rho, Sigma, T_mid, c_s2

    def density(self, r, z=0.):
        """
        The density function of the disk

        Parameters:
            r : float
                The radius inside the disk plane
            z : float
                The height above the disk

        Returns:
            out : float
                The density at the radius r
        """
        if isinstance(r, (np.ndarray, Sequence)):
            density = np.zeros(np.shape(r))
            rho = None; Sigma = None; T_mid = None; c_s2 = None
            for i in range(len(r)):
                rho, Sigma, T_mid, c_s2 = self.solve_eq(r[i], rho_0=rho, Sigma_0=Sigma, T_mid_0=T_mid, c_s2_0=c_s2)
                density[i] = rho
            return density

        return self.solve_eq(r)[0] * np.exp(-z**2 / self.scale_height(r)**2)

    def surface_density(self, r):
        """
        The surface density function of the disk

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The surface density at the radius r
        """
        if isinstance(r, (np.ndarray, Sequence)):
            surface_density = np.zeros(np.shape(r))
            rho = None; Sigma = None; T_mid = None; c_s2 = None
            for i in range(len(r)):
                rho, Sigma, T_mid, c_s2 = self.solve_eq(r[i], rho_0=rho, Sigma_0=Sigma, T_mid_0=T_mid, c_s2_0=c_s2)
                surface_density[i] = Sigma
            return surface_density

        return self.solve_eq(r)[1]


    def soundspeed(self, r):
        """
        The soundspeed c_s of the disk

        Parameters:
            r : float or array_like
                The radius at which to evaluate the soundspeed

        Returns:
            out : float or array_like (depending on r)
                The soundspeed at the radius r
        """
        if isinstance(r, (np.ndarray, Sequence)):
            c_s = np.zeros(np.shape(r))
            rho = None; Sigma = None; T_mid = None; c_s2 = None
            for i in range(len(r)):
                rho, Sigma, T_mid, c_s2 = self.solve_eq(r[i], rho_0=rho, Sigma_0=Sigma, T_mid_0=T_mid, c_s2_0=c_s2)
                c_s[i] = np.sqrt(c_s2)
            return c_s

        return np.sqrt(self.solve_eq(r)[3])


    def mach_number(self, r):
        """
        The mach number of the disk at radius r

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mach number

        Returns:
            out : float or array_like (depending on r)
                The mach number at the radius r
        """
        if isinstance(r, (np.ndarray, Sequence)):
            mach_number = np.zeros(np.shape(r))
            rho = None; Sigma = None; T_mid = None; c_s2 = None
            for i in range(len(r)):
                rho, Sigma, T_mid, c_s2 = self.solve_eq(r[i], rho_0=rho, Sigma_0=Sigma, T_mid_0=T_mid, c_s2_0=c_s2)
                h = Sigma/2./rho
                mach_number[i] = r[i]/h
            return mach_number

        rho, Sigma, T_mid, c_s2 = self.solve_eq(r)
        h = Sigma/2./rho
        return r/h

    def scale_height(self, r):
        """
        The disk scale height at radius r

        Parameters:
            r : float or array_like
                The radius at which to evaluate the scale height

        Returns:
            out : float or array_like (depending on r)
                The disk scale height at the radius r
        """
        if isinstance(r, (np.ndarray, Sequence)):
            h = np.zeros(np.shape(r))
            rho = None; Sigma = None; T_mid = None; c_s2 = None
            for i in range(len(r)):
                rho, Sigma, T_mid, c_s2 = self.solve_eq(r[i], rho_0=rho, Sigma_0=Sigma, T_mid_0=T_mid, c_s2_0=c_s2)
                h[i] = Sigma/2./rho
            return h

        rho, Sigma, T_mid, c_s2 = self.solve_eq(r)
        h = Sigma/2./rho
        return h

    def velocity(self, r, phi, z=0.):
        """
        The velocity of the particles in the disk

        Parameters:
            r : float
                The cylindrical radius inside the disk
            phi : float
                The cylindrical angle inside the disk
            z   : float
                The height above the disk plane

        Returns:
            out : tuple
                The velocity [v_X, v_Y, v_Z] in the fundamental plane
        """
        v_phi = np.sqrt(self.M/r)
        return  np.array([-v_phi*np.sin(phi) , v_phi*np.cos(phi), 0.])

    def optical_depth(self, r):
        """
        The optical depth of the disk

        Parameters:
            r : float or array_like
                The radius at which to evaluate the optical depth

        Returns:
            out : float or array_like (depending on r)
                The optical depth at the radius r
        """
        rho = self.density(r)
        T_mid = self.T_mid(r)
        kappa = self.opacity_scaling(rho/c.g_cm3_to_invpc2, T_mid) / c.g_cm2_to_invpc # TODO: check units
        Sigma = self.surface_density(r)
        return kappa*Sigma/2.


    def CreateInterpolatedHalo(self, r_grid):
        """
        Creates an InterpolatedHalo object of this instance for a given r_grid
        and adds the additional functions defined in this class

        Parameters:
            r_grid : array_like
                The grid in radius on which to interpolate the class functions

        Returns:
            out : InterpolatedHalo
                Instance of InterpolatedHalo with additional functions mimicking this class
        """
        res = []
        rho = None; Sigma = None; T_mid = None; c_s2 = None
        for i in range(len(r_grid)):
            rho, Sigma, T_mid, c_s2 = self.solve_eq(r_grid[i], rho_0=rho, Sigma_0=Sigma, T_mid_0=T_mid, c_s2_0=c_s2)
            # rho, Sigma, T_mid, c_s2 = self.solve_eq(r_grid[i])
            res.append([rho, Sigma, T_mid, c_s2])
        res = np.array(res)
        rho = res[:,0]; Sigma = res[:,1]; T_mid = res[:,2]; c_s = np.sqrt(res[:,3]);
        interpHalo = InterpolatedHalo(r_grid, rho, name=str(self))
        density_interp = interp1d(r_grid, rho, kind='linear', bounds_error=False, fill_value=(0.,0.))
        scale_height_interp = interp1d(r_grid, Sigma/2./rho, kind='linear', bounds_error=False, fill_value=(0.,0.))
        interpHalo.density = lambda r, z : density_interp(r)*np.exp(-z**2/scale_height_interp(r)**2)
        interpHalo.surface_density = interp1d(r_grid, Sigma, kind='linear', bounds_error=False, fill_value=(0.,0.))
        interpHalo.mach_number = interp1d(r_grid, r_grid/Sigma * 2 * rho, kind='linear', bounds_error=False, fill_value=(0.,0.))
        interpHalo.scale_height = scale_height_interp
        interpHalo.soundspeed = interp1d(r_grid, c_s, kind='linear', bounds_error=False, fill_value=(0.,0.))
        interpHalo.T_mid = interp1d(r_grid, T_mid, kind='linear', bounds_error=False, fill_value=(0.,0.))
        interpHalo.mass = interp1d(r_grid, BaryonicDisk.mass(interpHalo, r_grid), kind='linear', bounds_error=False, fill_value=(0.,0.))
        interpHalo.velocity = self.velocity
        interpHalo.Q = self.Q
        interpHalo.M = self.M
        interpHalo.alpha = self.alpha
        return interpHalo


    def __str__(self):
        return f"DerdzinskiMayerDisk (M = {self.M}, M_dot={self.M_dot}, alpha={self.alpha})"


class DoubleExponential(MatterHalo):
    """
    A class describing an axisymmetric, static, baryonic disk

    Attributes:
        r_min (float): A minimum radius below which the density is always 0, this is initialized to 0
        M_d   (float): The disk mass paramter
        R_d   (float): The disk scale length
        z_d   (float): The disk scale height
    """

    def __init__(self, M_d, R_d, z_d):
        """
        The constructor for the MiyamotoNagaiDisk class

        Parameters:
            M_d : float
                The disk mass paramter
            R_d : float
                The disk scale length
            z_d : float
                The disk scale height
        """
        self.M_d = M_d
        self.R_d = R_d
        self.z_d = z_d

    def density(self, r, z=0.):
        """
        The density function of the disk, see eq (6) of https://arxiv.org/pdf/2303.12838.pdf

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density
            z : float or array_like (optional)
                The height at which to evaluate the density (in cylindrical coordinates)

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r, height z
        """
        return self.M_d / 4. / np.pi / self.z_d / self.R_d**2  * np.exp(-r / self.R_d - np.abs(z) / self.z_d )

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return f"DoubleExponentialDisk: M_d={self.M_d}, R_d={self.R_d}, z_d={self.z_d}"


class TQMDisk(DerdzinskiMayerDisk):
    """
    The class describing a baryonic accretion disk as introduced by Thompson, Quataert & Murray
        in https://arxiv.org/pdf/astro-ph/0503027.pdf
        based on models of Shakura & Sunyaev and Sirko & Goodman, using an opacity description by Iglesias&Rogers, Alexander&Ferguson

    Due to the integral involved, this class is only used as an intermediary to create an InterpHalo object.

    Attributes:
        r_min (float): An minimum radius below which the density is always 0

        # These are the model parameters
        M     (float) : The mass of the central black hole
        M_dot_bh (float) : The accretion rate of the central black hole
        eps_star (float) : The star formation efficiency
        X (float) : The inflow velocity fraction of the speed of sound
    """

    def __init__(self, M, M_dot_bh, eps_star, X):
        """
        The constructor for the DerdzinskiMayerDisk class.

        Parameters:
            M : float
                The mass of the central black hole
            M_dot_bh : float
                The accretion rate of the central black hole
            eps_star : float
                The star formation efficiency
            X : float
                The infall velocity as a fraction of the speed of sound
        """
        BaryonicDisk.__init__(self)
        self.M = M
        self.M_dot_bh = M_dot_bh
        self.eps_star = eps_star
        self.X = X
        self.r_min = 6.*M

    ''' # TODO
    def opacity_scaling(self, rho, T):
        """
        Calculates the opacity scaling of an accretion disk depending on temperature and density

        Parameters:
            rho : floa
                The density in units of TODO
            T   : float
                The temperature in units of Kelvin

        Returns:
            out : float
                The Rosseland mean opacity
        """
        pass
    '''


    def solve_eq(self, r, r_last, M_dot_star_r_last, rho_0=None, Sigma_dot_0=None, T_mid_0=None, c_s2_0=None):
        """
        Solves the nonlinear equations of density, surface density, temperature and sound speed describing the disk
            as given in https://arxiv.org/pdf/2101.09146.pdf

        Parameters:
            r : float
                The radius of the point of interest
            r_last : float
                The radius of M_dot_star_r_last
            M_dot_star_r_last : float
                The value of M_dot_star, the mass loss due to star formation, at r_last -> integral is approximated in between
            *_0   (optional): float
                The initial guesses for the values of interest
                can e.g. be passed if a point nearby is known already

        Returns:
            rho: float
                The density at the radius r
            Sigma : float
                The surface density at radius r
            T_mid : float
                The midplane temperature
            c_s2   : float
                The soundspeed squared
            M_dot_star_r : float
                The new value of M_dot_star
        """
        if r <= self.r_min:
            return 0., 0., 0., 0.
        # calculate knowns at the given radius
        Omega = np.sqrt(self.M/r**3)

        M_dot_r_last = self.M_dot_bh + M_dot_star_r_last #integral over Sigma_dot until r_last
        Delta_r = r - r_last
        Q_min = 1.
        rho_max = Omega**2 / 2. / np.pi / Q_min

        # print(f"Omega = {Omega:.3e}, M_dot_prime={M_dot_prime:.3e}, T_eff={T_eff:.3e}")

        def f(x):
            sm_rho, ln_Sigma_dot, ln_T_mid, sm_c_s2 = x
            rho = rho_max/(1.+np.exp(-sm_rho)); Sigma_dot = np.exp(ln_Sigma_dot); T_mid = np.exp(ln_T_mid); c_s2 = 1./(1.+np.exp(-sm_c_s2))

            T_eff =  (( 3./8./np.pi * Omega**2 * self.M_dot_bh + 1./2.*self.eps_star * Sigma_dot) / self.stefan_boltzmann_constant)**(1./4.)
            kappa = self.opacity_scaling(rho/c.g_cm3_to_invpc2, T_mid) / c.g_cm2_to_invpc # TODO: check units
            h = np.sqrt(c_s2)/Omega
            Sigma = 2. * rho * h
            Q = Omega * np.sqrt(c_s2)  / np.pi / Sigma
            tau_opt = kappa*Sigma/2.
            p_gas = self.boltzmann_constant / self.hydrogen_mass * rho * T_mid
            p_rad = 1./2. * self.stefan_boltzmann_constant * tau_opt * T_eff**4
            p_tb = self.eps_star * Sigma_dot
            M_dot_r = self.X * np.sqrt(c_s2) * 2. * np.pi * r * Sigma

            # target values
            rho_t = (p_rad + p_gas + p_tb)/c_s2 if Q > 1.5*Q_min else rho_max*(1.-1e-10) # assert stability in outer parts
            Sigma_dot_t = 1./2./np.pi/r/Delta_r * (M_dot_r - M_dot_r_last) if Q <= 2.*Q_min else 1e-20
            c_s2_t = np.clip((p_rad + p_gas + p_tb) / rho, 1e-20, 1.-1e-10)
            T_mid_t = (3./8. * tau_opt + 1./2. + 1./4./tau_opt)**(1./4.) * T_eff

            target = np.array([np.log(rho_t/(rho_max-rho_t)) - sm_rho, np.log(Sigma_dot_t) - ln_Sigma_dot, np.log(T_mid_t) - ln_T_mid,  np.log(c_s2_t/(1.-c_s2_t)) - sm_c_s2])
            #print(f"rho = {rho}, Sigma_dot = {Sigma_dot}, T_mid={T_mid}, c_s2={c_s2}")
            #print(f"kappa = {kappa*c.g_cm2_to_invpc}, tau_opt = {tau_opt}, Sigma={Sigma}, h={h}, rho={rho/rho_max}")
            #print(f"rho={rho:.3e}->{rho_t:.3e}, Sigma_dot={Sigma_dot:.3e}->{Sigma_dot_t:.3e}, T_mid={T_mid:.3e}->{T_mid_t:.3e}, c_s2={c_s2:.3e}->{c_s2_t:.3e}")
            #print(target)
            return target

        # choose initial values
        T_eff_0 =  ( 3./8./np.pi * Omega**2 * self.M_dot_bh  / self.stefan_boltzmann_constant)**(1./4.)

        Sigma_dot_0 = 1e-20 if Sigma_dot_0 is None else Sigma_dot_0
        rho_0 = 1e-7*c.g_cm3_to_invpc2 if rho_0 is None else rho_0
        T_mid_0 = T_eff_0 if T_mid_0 is None else T_mid_0
        c_s2_0 = (self.boltzmann_constant / self.hydrogen_mass  * T_mid_0
                             +  4./3. * self.stefan_boltzmann_constant* T_mid_0**4 / rho_0)  if c_s2_0 is None else c_s2_0


        rho_0 = np.clip(rho_0, 1e-20, rho_max*(1.-1e-10))
        # compute solution
        x_0 = np.array([ np.log(rho_0/(rho_max-rho_0)), np.log(Sigma_dot_0), np.log(T_mid_0), np.log(c_s2_0/(1.-c_s2_0))])
        sol = root(f, x0 = x_0, method='lm', options={'xtol':1e-10,'ftol':1e-10, 'maxiter':1000})
        if not sol.success:
            print(sol.message, r, x_0, sol.x)

        rho, Sigma_dot, T_mid, c_s2 = np.exp(sol.x)
        rho = rho_max /(1. + 1./rho)
        c_s2 = 1./(1.+1./c_s2)

        M_dot_star = M_dot_star_r_last + 2.*np.pi * r * Delta_r * Sigma_dot
        return rho, Sigma_dot, T_mid, c_s2, M_dot_star


    def CreateInterpolatedHalo(self, r_grid):
        """
        Creates an InterpolatedHalo object of this instance for a given r_grid
        and adds the additional functions defined in this class

        Parameters:
            r_grid : array_like
                The grid in radius on which to interpolate the class functions

        Returns:
            out : InterpolatedHalo
                Instance of InterpolatedHalo with additional functions mimicking this class
        """
        res = []
        rho = None; Sigma_dot = None; T_mid = None; c_s2 = None; M_dot_star = 0.
        for i in range(len(r_grid)):
            rho, Sigma_dot, T_mid, c_s2, M_dot_star = self.solve_eq(r_grid[i], r_grid[i-1] if i > 0 else 0., M_dot_star, rho_0=rho, Sigma_dot_0=Sigma_dot, T_mid_0=T_mid, c_s2_0=c_s2)
            # rho, Sigma, T_mid, c_s2 = self.solve_eq(r_grid[i])
            res.append([rho, Sigma_dot, T_mid, c_s2])
        res = np.array(res)
        rho = res[:,0]; Sigma_dot = res[:,1]; T_mid = res[:,2]; c_s = np.sqrt(res[:,3]);
        Omega = np.sqrt(self.M/r_grid**3)
        Sigma = 2.*rho* c_s / Omega

        interpHalo = InterpolatedHalo(r_grid, rho, name=str(self))
        interpHalo.surface_density = interp1d(r_grid, Sigma, kind='linear', bounds_error=False, fill_value=(0.,0.))
        interpHalo.mach_number = interp1d(r_grid, r_grid/Sigma * 2 * rho, kind='linear', bounds_error=False, fill_value=(0.,0.))
        interpHalo.scale_height = interp1d(r_grid, Sigma/2./rho, kind='linear', bounds_error=False, fill_value=(0.,0.))
        interpHalo.soundspeed = interp1d(r_grid, c_s, kind='linear', bounds_error=False, fill_value=(0.,0.))
        interpHalo.T_mid = interp1d(r_grid, T_mid, kind='linear', bounds_error=False, fill_value=(0.,0.))
        interpHalo.Sigma_dot = interp1d(r_grid, Sigma_dot, kind='linear', bounds_error=False, fill_value=(0.,0.))
        interpHalo.mass = interp1d(r_grid, BaryonicDisk.mass(interpHalo, r_grid), kind='cubic', bounds_error=False, fill_value=(0.,0.))
        interpHalo.velocity = self.velocity
        interpHalo.M = self.M
        interpHalo.M_dot_bh = self.M_dot_bh
        interpHalo.eps_star = self.eps_star
        interpHalo.X = self.X
        return interpHalo

    # Only create an InterpHalo object
    def density(self, r):
        pass

    def surface_density(self, r):
        pass

    def soundspeed(self, r):
        pass

    def mach_number(self, r):
        pass

    def scale_height(self, r):
        pass

    def velocity(self, r):
        """
        The velocity of the particles in the disk

        Parameters:
            r : float or array_like
                The radius at which to evaluate the velocity

        Returns:
            out : tuple
                The velocity (v_r, v_phi) at the radius r
        """
        v_phi = np.sqrt(self.M/r)
        c_s = self.soundspeed(r)
        return  ( -self.X *c_s, v_phi)

    def __str__(self):
        return f"TQMDisk (M = {self.M}, M_dot_bh={self.M_dot_bh}, X={self.X}, eps_star={self.eps_star})"


class MiyamotoNagaiDisk(MatterHalo):
    """
    A class describing an axisymmetric, static, baryonic disk

    Attributes:
        r_min (float): A minimum radius below which the density is always 0, this is initialized to 0
        M_d   (float): The disk mass paramter
        R_d   (float): The disk scale length
        z_d   (float): The disk scale height
    """

    def __init__(self, M_d, R_d, z_d):
        """
        The constructor for the MiyamotoNagaiDisk class

        Parameters:
            M_d : float
                The disk mass paramter
            R_d : float
                The disk scale length
            z_d : float
                The disk scale height
        """
        self.M_d = M_d
        self.R_d = R_d
        self.z_d = z_d

    def density(self, r, z=0.):
        """
        The density function of the disk, see eq (2.69) of Binney&Tremain(2007)

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
        return f"MiyamotoNagaiDisk: M_d={self.M_d}, R_d={self.R_d}, z_d={self.z_d}"




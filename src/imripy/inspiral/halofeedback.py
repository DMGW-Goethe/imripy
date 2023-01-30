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
    b_min = 15e3 * c.m_to_pc   # 15 km in pc
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

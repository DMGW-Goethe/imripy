import numpy as np
import torch
import torchsde

import time
import imripy.constants as c
from . import forces
from .classic import *


class Stochastic:

    class EvolutionOptions(Classic.EvolutionOptions):
        """
        This class allows to modify the behavior of the evolution of the differential equations

        Attributes:
            accuracy : float
                An accuracy parameter that is passed to solve_ivp
            verbose : int
                A verbosity parameter ranging from 0 to 2
            elliptic : bool
                Whether to model the inspiral on eccentric orbits, is set automatically depending on e0 passed to Evolve
            m2_chance : bool
                Whether to evolve the secondary's mass m2 with time (e.g. through accretion)
            dissipativeForces : list of forces.DissipativeForces
                List of the dissipative forces employed during the inspiral
            stochasticForces : list of forces.StochasticForces
                List of the stochastic forces employed during the inspiral
            ignoreStochasticContribution : bool
                If True, the dW contribution will be set to zero in the SDE evolution - Primarily for testing purposes
            gwEmissionLoss : bool
                These parameters are for backwards compatibility - Only applies if dissipativeForces=None - then forces.GWLoss is added to the list
            dynamicalFrictionLoss : bool
                These parameters are for backwards compatibility - Only applies if dissipativeForces=None - then forces.DynamicalFriction is added to the list
            **kwargs : additional parameter
                Will be saved in opt.additionalParameters and will be available throughout the integration

        """
        def __init__(self, accuracy=1e-10, verbose=1, elliptic=True, m2_change=False,
                                    dissipativeForces=None, gwEmissionLoss = True, dynamicalFrictionLoss = True,
                                    stochasticForces=None, ignoreStochasticContribution = False, adaptiveStepSize=False,
                                    considerRelativeVelocities=False, progradeRotation = True,
                                    **kwargs):
            super().__init__(accuracy=accuracy, verbose=verbose, elliptic=elliptic, m2_change=m2_change,
                                dissipativeForces=dissipativeForces, gwEmissionLoss=gwEmissionLoss, dynamicalFrictionLoss=dynamicalFrictionLoss,
                                considerRelativeVelocities=considerRelativeVelocities, progradeRotation=progradeRotation,
                                **kwargs)
            # Overwrite changes made by parent class
            self.dissipativeForces = dissipativeForces
            self.stochasticForces = stochasticForces
            self.ignoreStochasticContribution = ignoreStochasticContribution
            self.adaptiveStepSize = adaptiveStepSize




    def dEdL_diffusion(hs, ko, opt=EvolutionOptions()):
        """
        The function gives the diffusion matrix of the SDE for the  energy and angular momentum
        The variances are on the diagonal and the covariance on the off-diagonal
        Calls each stochastic force and gets its diffusion matrix. For normally distributed noise, the (co)variances are additive.

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The diffusion matrix of the SDE
        """
        dEdL_cov_tot = np.zeros(shape=(2,2))
        dEdL_cov_out = ""
        for df in opt.stochasticForces:
            dEdL_cov = df.dEdL_diffusion(hs, ko, opt)
            dEdL_cov_tot += dEdL_cov
            if opt.verbose > 2:
                dEdL_cov_out += f"{df.name}:{dEdL_cov}, "

        if opt.verbose > 2:
            print(dEdL_cov_out)
        return dEdL_cov_tot



    def dade_diffusion(hs, ko, da_dt, de_dt, opt=EvolutionOptions()):
        """
        This function gives the diffusion matrix of the SDE for the semimajor axis and eccentricity
        by taking the diffusion matrix for E and L and transforming it to a and e with the Jacobian.
        It transforms the time derivative according to the Ito formula

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the current orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The diffusion matrix of the SDE
        """
        a = ko.a; e = ko.e
        E = Classic.E_orbit(hs, ko, opt)
        L = Classic.L_orbit(hs, ko, opt)
        if opt.verbose > 3:
            print(rf"a={a:.3e}, e={e:.3e}, E={E:.3e}, L={L:.3e}")

        dade_dEdL = -np.array([[- a / E, 0.],
                              [ (e**2 - 1)/ (2.*e*E) , (e**2 - 1)/ (e*L)]] ) # The Jacobian del(a,e)/del(E,L)

        sigma = Stochastic.dEdL_diffusion(hs, ko, opt=opt)

        sigma_prime = np.matmul(dade_dEdL, sigma) # matrix product
        if opt.verbose > 3:
            print(rf"dade_dEdL={dade_dEdL}, sigma={sigma}, sigma_prime={sigma_prime}")

        D = np.matmul(sigma, sigma.T)

        pref = 2./ko.m_red**3 / ko.m_tot**2
        Ha = -np.array([[2.*a/E**2, 0.], [0., 0.]])  # The Hessian of a
        He = -np.array([[ - pref**2 * L**4 / 4. / e**3, pref*L * (e**2 + 1.) / 2. / e**3],
                        [0., pref * E / e**3]])
        He[1,0] = He[0,1]

        da_dt_prime = da_dt + np.sum(D*Ha) / 2. # elementwise multiplication
        de_dt_prime = de_dt + np.sum(D*He) / 2.

        if opt.verbose > 3:
            print(rf"D={D}, Ha={Ha}, D*Ha={D*Ha}, np.sum(D*Ha)={np.sum(D*Ha)}")
            print(rf"He={He}, D*He={D*He}, np.sum(D*He)={np.sum(D*He)}")

        return da_dt_prime, de_dt_prime, sigma_prime




    def handle_args(hs, ko, a_fin, t_0, t_fin, opt):
        opt.elliptic = ko.e > 0.

        # calculate relevant timescales
        def g(e):
            return e**(12./19.)/(1. - e**2) * (1. + 121./304. * e**2)**(870./2299.)

        t_coal =  5./256. * ko.a**4 / ko.m_tot**2 / ko.m_red
        if opt.elliptic:
            t_coal = t_coal * 48./19. / g(ko.e)**4 * quad(lambda e: g(e)**4 *(1-e**2)**(5./2.) /e/(1. + 121./304. * e**2), 0., ko.e, limit=100)[0]   # The inspiral time according to Maggiore (2007)

        if t_fin is None:
            t_fin = 1.2 * t_coal *( 1. - a_fin**4 / ko.a**4)    # This is the time it takes with just gravitational wave emission

        if a_fin == 0.:
            a_fin = hs.r_isco     # Stop evolution at r_isco

        ko = copy.deepcopy(ko) # to avoid changing the passed object
        ko.prograde = opt.progradeRotation # Check?
        return hs, ko, a_fin, t_0, t_fin, opt


    def Evolve(hs, ko, a_fin=0., t_0=0., t_fin=None, opt=EvolutionOptions(),
               batch_size=1):
        """
        The function evolves the SDE of the inspiraling system.
        The host system defines the central MBH and environment, the Keplerian Orbit the secondary on its orbit.
        The secondary mass, and its keplerian parameters can be evolved in time.
        The dissipative and stochastic forces are part of the EvolutionOptions object.

        Parameters:
            hs (HostSystem) : The host system object
            ko (KeplerOrbit): The Kepler orbit object describing the initial orbit
            a_fin (float)   : The semimajor axis at which to stop evolution
            t_0    (float)  : The initial time
            t_fin  (float)  : The time until the system should be evolved, if None then the estimated coalescence time will be used
            opt   (EvolutionOptions) : Collecting the options for the evolution of the differential equations
            batch_size (int): Number of concurrent integrations

        Returns:
            ev : EvolutionObject or list of EvolutionObjects
                An evolution object that contains the results - or a list of size batch_size
        """
        hs, ko, a_fin, t_0, t_fin, opt = Stochastic.handle_args(hs, ko, a_fin, t_0, t_fin, opt)


        # no rescale
        state_size = 2  # we integrate a, e
        #a_scale = ko.a
        a_scale = 1.
        #t_scale = t_fin
        t_scale = 1.

        class SDE(torchsde.SDEIto):

            class HitBlackHoleEvent(torchsde.HitTargetEvent):
                def __init__(self):
                    super(SDE.HitBlackHoleEvent, self).__init__(terminal=True)

                def target(self, t, y):
                    a = y[:,0]
                    e = y[:,1]
                    return  a*a_scale*(1.-e) - 4*hs.r_schwarzschild

            class HitAfinEvent(torchsde.HitTargetEvent):
                def __init__(self):
                    super(SDE.HitAfinEvent, self).__init__(terminal=True)

                def target(self, t, y):
                    a = y[:,0]
                    return  a*a_scale - a_fin

            def __init__(self):
                super().__init__(noise_type = 'general')
                self.hs = hs
                self.opt = opt


            def dy(self, t, y):
                """
                This function is like the dy_dt in the classic case, it takes a single set of a, e and calculates dy_dt and dy_dW

                Parameters:
                    t : float
                        The (scaled) time variable (isn't generally used)
                    y : np.array (1D)
                        Contains the current integration state for a single iteration of a, e

                Returns
                    dy_dt : np.array (1D)
                        The drift term of the SDE
                    dy_dW : np.array (2D)
                        The diffusion term of the SDE
                """
                t = t*t_scale
                # Unpack array
                ko.a, ko.e = np.array(y)
                ko.a *= a_scale

                if opt.verbose > 2:
                    tic = time.perf_counter()

                da_dt, dE_dt = Classic.da_dt(self.hs, ko, opt=self.opt, return_dE_dt=True)
                de_dt = Classic.de_dt(self.hs, ko, dE_dt=dE_dt, opt=self.opt) if self.opt.elliptic else 0.

                da_dt, de_dt, dy_dW = Stochastic.dade_diffusion(hs, ko, da_dt, de_dt, opt=opt)

                if self.opt.verbose > 2:
                    toc = time.perf_counter()
                    print(rf"dy: t={t : 0.5e} ({t/t_scale:0.3e} t_scale), a={a : 0.3e}({a/hs.r_isco : 0.3e} r_isco)({a/a_fin:0.3e} a_fin), da/dt={da_dt : 0.3e}, da/dW={dy_dW[0,0]:0.3e}\\n"
                          +  rf"\\t e={e : 0.3e}, de/dt={ de_dt : 0.3e}, de/dW={dy_dW[1,0] + dy_dW[1,1] : 0.3e}\\n"
                           + rf"\\t elapsed real time: { toc-tic :0.4f} s")

                dy_dW[0,0]/= a_scale
                if self.opt.ignoreStochasticContribution:
                    dy_dW = np.zeros(np.shape(dy_dW))
                dy_dt = np.array([[da_dt/a_scale, de_dt]])
                return dy_dt * t_scale, dy_dW*np.sqrt(t_scale)   # TODO : Check scaling


            def f_and_g(self, t, y):
                """
                This is a wrapper function for torchsde and has two purposes:
                    - Get out of the torch.Tensors and convert to np.array for the rest of the code to work
                    - for batch_size > 1 it takes the tensors apart and calls the calc_derivatives function individually
                        This loses a lot of the speed but improving this would necessitate a total rewrite
                TODO:
                    Multithreading?
                Parameters:
                    t : torch.Tensor
                        The time of the integration
                    y : torch.Tensor
                        The state of the integration

                Returns:
                    f : torch.Tensor
                        The drift of the integration
                    g : torch.Tensor
                        The diffusion of the integration
                """
                f = torch.Tensor(size=y.shape)
                g = torch.Tensor(size=(*y.shape, state_size))
                for i, ys in enumerate(y):
                    dy_dt, dy_dW = self.dy(t.item(), np.array(ys))
                    f[i] = torch.Tensor(dy_dt)
                    g[i] = torch.Tensor(dy_dW)

                if self.opt.verbose > 2:
                    print("f_and_g: ", t, y, f, g)
                return f, g

        # Initial conditions
        y0 = torch.full(size=(batch_size, state_size), fill_value=0.)
        for batch in range(y0.shape[0]):
            y0[batch] = torch.tensor([ko.a / a_scale, ko.e])

        # tspan
        tspan = [0., t_fin/t_scale]
        dt_min = t_fin/t_scale * np.min([1e-7, opt.accuracy])
        dt = 1e3*dt_min

        # Events:
        events = [SDE.HitBlackHoleEvent(), SDE.HitAfinEvent()]
        if opt.verbose > 1:
            events.append(torchsde.VerboseEvent(10, avg=True))
        if not opt.additionalEvents is None:
            for ev in opt.additionalEvents:
                events.append(ev)

        # Output
        if opt.verbose > 0:
            print("Evolving from ", ko.a/hs.r_isco, " to ", a_fin/hs.r_isco,"r_isco ", ("with initial eccentricity " + str(ko.e)) if opt.elliptic else " on circular orbits", " with ", opt)
            if opt.verbose > 1:
                print(rf"Initial stepsize: {dt:.2e}, Minimal Stepsize: {dt_min:.2e}, Maximal_time: {tspan[1]:.2e}")

        # Evolve
        tic = time.perf_counter()
        sde = SDE()
        with torch.no_grad():
            # adaptive
            ts, sol, events = torchsde.solve_sde(sde, y0, tspan, method='euler', events=events, adaptive=opt.adaptiveStepSize,
                                                            rtol=opt.accuracy, atol=opt.accuracy, dt_min=dt_min, dt=dt)

        toc = time.perf_counter()

        # Collect results
        evs = []
        for batch in range(sol.shape[1]):
            # find if and when the integration artificially terminated
            t = np.unique(np.array(ts)[:, batch])*t_scale
            a = a_scale * np.array(sol[:len(t), batch, 0])
            ev = Classic.EvolutionResults(hs, opt, t, ko.m2, a, e=np.array(sol[:len(t), batch, 1]) if opt.elliptic else np.zeros(np.shape(t)),
                                            periapse_angle=ko.periapse_angle, inclination_angle=ko.inclination_angle, longitude_an=ko.longitude_an)
            evs.append(ev)

        if opt.verbose > 0:
            print(f" -> Evolution took {toc-tic:.4f}s")

        return evs if len(evs) > 1 else evs[0]


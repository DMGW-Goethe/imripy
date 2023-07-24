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
            gwEmissionLoss : bool
                These parameters are for backwards compatibility - Only applies if dissipativeForces=None - then forces.GWLoss is added to the list
            dynamicalFrictionLoss : bool
                These parameters are for backwards compatibility - Only applies if dissipativeForces=None - then forces.GWLoss is added to the list
            **kwargs : additional parameter
                Will be saved in opt.additionalParameters and will be available throughout the integration

        """
        def __init__(self, accuracy=1e-10, verbose=1, elliptic=True, m2_change=False,
                                    dissipativeForces=None, gwEmissionLoss = True, dynamicalFrictionLoss = True,
                                    stochasticForces=None,
                                    considerRelativeVelocities=False, progradeRotation = True,
                                    **kwargs):
            super().__init__(accuracy=accuracy, verbose=verbose, elliptic=elliptic, m2_change=m2_change,
                                dissipativeForces=dissipativeForces, gwEmissionLoss=gwEmissionLoss, dynamicalFrictionLoss=dynamicalFrictionLoss,
                                considerRelativeVelocities=considerRelativeVelocities, progradeRotation=progradeRotation,
                                **kwargs)
            self.stochasticForces = stochasticForces or []


        def __str__(self):
            s = Classic.EvolutionOptions.__str__(self)
            s += " - Stochastic Forces: {"
            for sf in self.stochasticForces:
                s += str(sf) + ", "
            s += "}"
            return s

    def dE_dW(sp, a, e=0., opt=EvolutionOptions()):
        """
        The function gives the stochastic energy loss of the secondary object

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The stochastic energy loss
        """
        dE_dW_tot = 0.
        dE_dW_out = ""
        for df in opt.stochasticForces:
            dE_dW = df.dE_dW(sp, a, e, opt)
            dE_dW_tot += dE_dW
            if opt.verbose > 2:
                dE_dW_out += f"{df.name}:{dE_dW}, "

        if opt.verbose > 2:
            print(dE_dW_out)
        return  dE_dW_tot


    def dL_dW(sp, a, e, opt=EvolutionOptions()):
        """
        The function gives the stochastic angular momentum loss of the secondary object

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The stochastic angular momentum loss
        """
        dL_dW_tot = 0.
        dL_dW_out = ""
        for df in opt.stochasticForces:
            dL_dW = df.dL_dW(sp, a, e, opt)
            dL_dW_tot += dL_dW
            if opt.verbose > 2:
                dL_dW_out += f"{df.name}:{dL_dW}, "

        if opt.verbose > 2:
            print(dL_dW_out)
        return  dL_dW_tot


    def da_dW(sp, a, e=0., opt=EvolutionOptions, return_dE_dW=False):
        """
        The function gives the diffusion term for the semimajor axis

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations
            return_dE_dW (bool)    : Whether to return dE_dW in addition to da_dt, to save computation time later

        Returns:
            da_dW : float
                The diffusion term for the semimajor axis
            dE_dW : float
                The diffusion term for the orbital energy
        """
        dE_dW = Stochastic.dE_dW(sp, a, e, opt)

        dE_orbit_da = Classic.dE_orbit_da(sp, a, e, opt)

        if return_dE_dW:
            return dE_dW / dE_orbit_da, dE_dW

        return    ( dE_dW / dE_orbit_da )


    def de_dW(sp, a, e, dE_dW=None, opt=EvolutionOptions()):
        """
        The function gives the diffusion term for the eccentricity

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            dE_dW (float)   : Optionally, the dE_dW value if it was computed previously
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            de_dW : float
                The diffusion term for the eccentricity
        """
        if e <= 0. or not opt.elliptic:
            return 0.

        dE_dW = Stochastic.dE_dW(sp, a, e, opt) if dE_dW is None else dE_dW
        E = Classic.E_orbit(sp, a, e, opt)
        dL_dW = Stochastic.dL_dW(sp, a, e, opt)
        L = Classic.L_orbit(sp, a, e, opt)

        if opt.verbose > 2:
            print("dE_dW/E=", dE_dW/E, "2dL_dW/L=", 2.*dL_dW/L, "diff=", dE_dW/E + 2.*dL_dW/L )

        return - (1.-e**2)/2./e *(  dE_dW/E + 2. * dL_dW/L   )



    def Evolve(sp, a_0, e_0=0., a_fin=0., t_0=0., t_fin=None, opt=EvolutionOptions(),
               batch_size=1, t_size = 100):
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
            batch_size (int): Number of concurrent integrations
            t_size (int)    : The initial size of the grid in time

        Returns:
            ev : EvolutionObject or list of EvolutionObjects
                An evolution object that contains the results - or a list of size batch_size
        """
        opt.elliptic = e_0 > 0.

        # calculate relevant timescales
        def g(e):
            return e**(12./19.)/(1. - e**2) * (1. + 121./304. * e**2)**(870./2299.)

        t_coal =  5./256. * a_0**4/sp.m_total()**2 /sp.m_reduced()
        if opt.elliptic:
            t_coal = t_coal * 48./19. / g(e_0)**4 * quad(lambda e: g(e)**4 *(1-e**2)**(5./2.) /e/(1. + 121./304. * e**2), 0., e_0, limit=100)[0]   # The inspiral time according to Maggiore (2007)

        if t_fin is None:
            t_fin = t_coal *( 1. - a_fin**4 / a_0**4)    # This is the time it takes with just gravitational wave emission

        if a_fin == 0.:
            a_fin = sp.r_isco()     # Stop evolution at r_isco

        # set scales to get rescale integration variables
        a_scale = a_0
        t_scale = t_fin
        m_scale = sp.m2 if opt.m2_change else 1.
        #torch.set_default_tensor_type(torch.DoubleTensor)

        if opt.verbose > 0:
            print("Evolving from ", a_0/sp.r_isco(), " to ", a_fin/sp.r_isco(),"r_isco ", ("with initial eccentricity " + str(e_0)) if opt.elliptic else " on circular orbits", " with ", opt)

        class SDE(torchsde.SDEStratonovich):

            def __init__(self):
                super().__init__(noise_type = 'diagonal')  # TODO : Implement covariance
                self.sp = sp
                self.opt = opt


            def dy(self, t, y):
                """
                This function is like the dy_dt in the classic case, it takes a single set of a, e, m2 and calculates dy_dt and dy_dW

                Parameters:
                    t : float
                        The (scaled) time variable (isn't generally used)
                    y : np.array (1D!)
                        Contains the current integration state for a single iteration of a, e, m2

                Returns
                    dy_dt : np.array (1D)
                        The drift term of the SDE
                    dy_dW : np.array (1D)
                        The diffusion term of the SDE
                """
                t = t*t_scale
                # Unpack array
                a, e, m2 = np.array(y)
                a *= a_scale; sp.m2 = m2 * m_scale if self.opt.m2_change else sp.m2

                if opt.verbose > 1:
                    tic = time.perf_counter()

                da_dt, dE_dt = Classic.da_dt(self.sp, a, e, opt=self.opt, return_dE_dt=True)
                de_dt = Classic.de_dt(self.sp, a, e, dE_dt=dE_dt, opt=self.opt) if self.opt.elliptic else 0.
                dm2_dt = Classic.dm2_dt(self.sp, a, e, self.opt) if self.opt.m2_change else 0.

                da_dW, dE_dW = Stochastic.da_dW(self.sp, a, e, opt=self.opt, return_dE_dW=True)
                de_dW = Stochastic.de_dW(self.sp, a, e, dE_dW=dE_dW, opt=self.opt) if self.opt.elliptic else 0.
                dm2_dW = 0.

                if self.opt.verbose > 1:
                    toc = time.perf_counter()
                    print(rf"Step: t={t : 0.1e}, a={a : 0.1e}({a/sp.r_isco() : 0.1e} r_isco), da/dt={da_dt : 0.1e}, da/dW={da_dW:0.1e}\\n"
                          +  rf"\\t e={e : 0.1e}, de/dt={ de_dt : 0.1e}, de/dW={de_dW : 0.1e}, m2={sp.m2 : 0.1e} ({sp.m2/c.solar_mass_to_pc : 0.1e} solar mass), dm2/dt={dm2_dt : 0.1e} \\n"
                           + rf"\\t elapsed real time: { toc-tic } s")

                dy_dt = np.array([[da_dt/a_scale, de_dt, dm2_dt/m_scale]])
                dy_dW = np.array([[da_dW/a_scale, de_dW, dm2_dW/m_scale]])
                return dy_dt * t_scale, dy_dW*np.sqrt(t_scale)   # TODO : Check scaling

            def termination_condition(self, t, y, verbose = False):
                """
                Gives the termination condition for falling into a black hole
                """
                a, e, m2 = y
                #if verbose:
                #    print(t, y, a, a*a_scale, a*a_scale/sp.r_isco(), sp.r_isco(), a*a_scale < sp.r_isco())
                return a*a_scale < sp.r_isco() or a*a_scale*(1.-e) < sp.r_schwarzschild()

            def f_and_g(self, t, y):
                """
                This is a wrapper function for torchsde and has three purposes:
                    - Get out of the torch.Tensors and convert to np.array for the rest of the code to work
                    - there is no event detection like for (scipy->)solve_ivp, instead we have to provide a grid in t to sdeint. Therefore, we have to
                        detect the infall into the black hole and stop the integration artificially. In this case zeros are returned for both drift and diffusion,
                        effectively freezing the integration. These have to be trimmed out later.
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
                g = torch.Tensor(size=y.shape)
                for i, ys in enumerate(y):
                    if self.termination_condition(t.item(), np.array(ys)):
                        f[i] = torch.zeros(size=ys.shape)
                        g[i] = torch.zeros(size=ys.shape)
                        continue
                    dy_dt, dy_dW = self.dy(t.item(), np.array(ys))
                    f[i] = torch.Tensor(dy_dt)
                    g[i] = torch.Tensor(dy_dW)

                if self.opt.verbose > 1:
                    print("Step: ", t, y, f, g)
                return f, g


        # Initial conditions
        state_size = 3
        y0 = torch.full(size=(batch_size, state_size), fill_value=0.)
        for batch in range(y0.shape[0]):
            y0[batch] = torch.tensor([a_0 / a_scale, e_0, sp.m2/m_scale])

        # tspan
        ts = torch.linspace(0., t_fin/t_scale, t_size)

        # Evolve
        tic = time.perf_counter()
        sde = SDE()
        with torch.no_grad():
            sol = torchsde.sdeint(sde, y0, ts, adaptive=True, rtol=opt.accuracy, atol=opt.accuracy)
        toc = time.perf_counter()

        # Collect results
        evs = []
        for batch in range(sol.shape[1]):
            # find if and when the integration artificially terminated
            terminated = 0
            while terminated < sol.shape[0] and not sde.termination_condition(ts[terminated].item(), np.array(sol[terminated, batch, :]), verbose=True) :
                terminated += 1
            terminated += 1 # Make sure to get the first point inside the black hole
            t = np.array(ts)[:terminated]*t_scale
            a = a_scale * np.array(sol[:terminated, batch, 0])
            ev = Classic.EvolutionResults(sp, opt, t, a)
            ev.e = np.array(sol[:terminated, batch, 1]) if opt.elliptic else np.zeros(np.shape(ev.t))
            ev.m2 = m_scale*np.array(sol[:terminated,batch,2]) if opt.m2_change else sp.m2
            evs.append(ev)

        if opt.verbose > 0:
            print(f" -> Evolution took {toc-tic:.4f}s")

        return evs if len(evs) > 1 else evs[0]

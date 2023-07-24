import numpy as np
import torch
from torch import nn
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
        The function gives the total energy loss of the orbiting small black hole due to the dissipative effects
           on a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The total energy loss
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
        The function gives the total angular momentum loss of the secondary object
            on a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The total angular momentum loss
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

        """
        dE_dW = Stochastic.dE_dW(sp, a, e, opt)

        dE_orbit_da = Classic.dE_orbit_da(sp, a, e, opt)

        if return_dE_dW:
            return dE_dW / dE_orbit_da, dE_dW

        return    ( dE_dW / dE_orbit_da )


    def de_dW(sp, a, e, dE_dW=None, opt=EvolutionOptions()):
        """

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



    def Evolve(sp, a_0, e_0=0., a_fin=0., t_0=0., t_fin=None, opt=EvolutionOptions()):
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

        Returns:
            ev : Evolution
                An evolution object that contains the results
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

        t_step_max = np.inf
        if opt.verbose > 0:
            print("Evolving from ", a_0/sp.r_isco(), " to ", a_fin/sp.r_isco(),"r_isco ", ("with initial eccentricity " + str(e_0)) if opt.elliptic else " on circular orbits", " with ", opt)

        class SDE(nn.Module):

            def __init__(self):
                super().__init__()
                self.noise_type = 'diagonal'
                self.sde_type = 'stratonovich'
                self.sp = sp
                self.opt = opt


            # Define the (drift) evolution function
            def f(self, t, y):
                t = t.item()*t_scale

                # Unpack array
                a, e, m2 = np.array(y[0])
                a *= a_scale; sp.m2 = m2 * m_scale if self.opt.m2_change else sp.m2

                if opt.verbose > 1:
                    tic = time.perf_counter()

                da_dt, dE_dt = Classic.da_dt(self.sp, a, e, opt=self.opt, return_dE_dt=True)
                de_dt = Classic.de_dt(self.sp, a, e, dE_dt=dE_dt, opt=self.opt) if self.opt.elliptic else 0.
                dm2_dt = Classic.dm2_dt(self.sp, a, e, self.opt) if self.opt.m2_change else 0.

                if self.opt.verbose > 1:
                    toc = time.perf_counter()
                    print("classic step: t=", t, "a=", a, "da/dt=", da_dt, "e=", e, "de/dt=", de_dt, "m2=", self.sp.m2, "dm2_dt=", dm2_dt,
                            " elapsed real time: ", toc-tic)

                dy = torch.tensor([[da_dt/a_scale, de_dt, dm2_dt/m_scale]])
                return dy * t_scale

            # Define the stochastic component
            def g(self, t, y):
                t = t.item()*t_scale

                # Unpack array
                a, e, m2 = np.array(y[0])
                a *= a_scale; sp.m2 = m2 * m_scale if self.opt.m2_change else sp.m2

                if opt.verbose > 1:
                    tic = time.perf_counter()

                da_dW, dE_dW = Stochastic.da_dW(self.sp, a, e, opt=self.opt, return_dE_dW=True)
                de_dW = Stochastic.de_dW(self.sp, a, e, dE_dW=dE_dW, opt=self.opt) if self.opt.elliptic else 0.
                dm2_dW = 0.

                if self.opt.verbose > 1:
                    toc = time.perf_counter()
                    print("stochastic step: t=", t, "a=", a, "da/dW=", da_dW, "e=", e, "de/dW=", de_dW, "m2=", sp.m2, "dm2_dW=", dm2_dW,
                        " elapsed real time: ", toc-tic)

                dy = torch.tensor([[da_dW/a_scale, de_dW, dm2_dW/m_scale]])
                return dy * t_scale

        # Termination condition
        fin_reached = lambda t,y, *args: y[0] - a_fin/a_scale
        fin_reached.terminal = True

        batch_size, state_size, t_size = 1, 3, 100

        # Initial conditions
        y0 = torch.full(size=(batch_size, state_size), fill_value=0.)
        y0[0] = torch.tensor([a_0 / a_scale, e_0, sp.m2/m_scale])

        # tspan
        ts = torch.linspace(0, t_fin/t_scale, t_size)

        # Evolve
        tic = time.perf_counter()
        sde = SDE()
        with torch.no_grad():
            ys = torchsde.sdeint(sde, y0, ts, adaptive=True, rtol=opt.accuracy, atol=opt.accuracy)  
        toc = time.perf_counter()

        # Collect results
        ys = ys.cpu().squeeze()
        t = np.array(ts.cpu())*t_scale
        a = a_scale * np.array(ys)[:,0]
        ev = Classic.EvolutionResults(sp, opt, t, a)
        ev.e = np.array(ys)[:,1] if opt.elliptic else np.zeros(np.shape(ev.t))
        ev.m2 = m_scale*np.array(ys)[:,2] if opt.m2_change else sp.m2;

        if opt.verbose > 0:
            print(f" -> Evolution took {toc-tic:.4f}s")

        return ev

import numpy as np

from mpi4py import MPI
import datetime

from imripy import constants as c, merger_system as ms, halo, kepler, inspiral
import torchsde
import common

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 100
acc = 5e-11
print(size, rank, n, " initialized")

m1 = 1e5 * c.solar_mass_to_pc
m2 = 1.  * c.solar_mass_to_pc

hs = ms.HostSystem(m1)

# Stellar Halo
stellarHalo, stellarDiffusion = common.StellarDistribution(hs, m2)


opt = inspiral.Stochastic.EvolutionOptions(dissipativeForces=[inspiral.forces.GWLoss()],
                                            stochasticForces=[stellarDiffusion],
                                            verbose = 1, accuracy=acc, adaptiveStepSize=False)


class InspiralEvent(torchsde.HitTargetEvent):
    def __init__(self):
        super(InspiralEvent, self).__init__(terminal=True)

    def target(self, t, y):
        a = y[:,0]
        return a - 1e2*hs.r_isco

opt.additionalEvents = [InspiralEvent()]

# Initial orbi
a0s = np.array([1e3, 5e3, 1e4, 5e4, 1e5, 1e6, 1e7])*hs.r_isco
for a0 in a0s:
    print(rank, a0)
    #e0 = np.clip(1. - 1e3*8. * m1 / a0 , 0.7, 1.)
    e0 = 0.1
    ko = kepler.KeplerOrbit(hs, m2, a0, e0)

    # Estimate inspiral time
    t_GW = 3.*2**(7./2)/85 * ko.a**4 * (1-ko.e)**(7./2) / m1**2 / m2
    k = 0.34
    sigma = np.sqrt(m1 / (1.+ stellarHalo.alpha) / ko.a)
    t_AM =  ( 2*k*sigma**3 / (stellarHalo.density(ko.a)/stellarDiffusion.E_m_s)
          / stellarDiffusion.E_m_s**2 / stellarDiffusion.CoulombLogarithm * (1.-ko.e))


    for i in range(n//size):
        tspan = 1e1*np.min([t_GW,t_AM])
        ev_st = inspiral.Stochastic.Evolve(hs, ko, opt=opt, batch_size =1, t_fin = tspan)
        t_ev = ev_st.t[-1]

        d = datetime.datetime.now()
        filename = f"inspirals/sd/sd_{a0/hs.r_isco:.1e}_{rank}_{i}_{d.timestamp()}.npz" # timestamp in case filename exists
        ev_st.save(filename)
        print(d, ":  ", rank, "saved ", filename)



import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from imripy import halo
from imripy import merger_system as ms
from imripy import inspiral


m1 = 1e3 * ms.solar_mass_to_pc
m2 = 1. * ms.solar_mass_to_pc
D = 5e8  # in pc

sp_0 = ms.SystemProp(m1, m2, halo.ConstHalo(0.), D)

rho6 = 5.448e15 * ms.solar_mass_to_pc  # in 1/pc^2
alpha = 7./3.
sp_dm_stat = ms.SystemProp(m1, m2, halo.Spike.FromRho6(rho6, m1, alpha), D)

def makeDynamicSpike(sp_stat):
    extPotential = lambda r:sp_stat.m1/r
    r_grid = np.geomspace(1e-1*sp_stat.r_isco(), 1e8*sp_stat.r_isco(), 100)
    Eps_grid = np.geomspace(extPotential(r_grid[-1]), extPotential(r_grid[0]), 500)
    dynSpike = halo.DynamicSS.FromSpike(Eps_grid, sp_stat, sp_stat.halo)
    sp_dyn = ms.SystemProp(sp_stat.m1, sp_stat.m2, dynSpike, sp_stat.D)
    return sp_dyn

sp_dm_dyn = makeDynamicSpike(sp_dm_stat)

def plotEvolution(sp, ev, ax_a=None, ax_e=None, label="", ax_ae=None, ax_m=None, m2=1.):
    l = None
    if not ax_a is None:
        l, = ax_a.loglog(ev.t/ms.year_to_pc, ev.a/sp.r_isco(), label=label)
    if not ax_e is None:
        l, = ax_e.loglog(ev.t/ms.year_to_pc, ev.e, color=(l.get_c() if not l is None else None), label=label)
    if not ax_m is None:
        l, = ax_m.loglog(ev.t/ms.year_to_pc, ev.m2/m2-1., linestyle='--', color=(l.get_c() if not l is None else None), label=label)
    if not ax_ae is None:
        l, = ax_ae.plot(ev.a/sp.r_isco(), ev.e, color=(l.get_c() if not l is None else None), label=label)
    return l

def compareAccretionModels(sp_0, sp_dm, a0, e0, ax_a=None, ax_e=None, ax_ae=None, ax_m=None, label="", acc=1e-8, verbose=1, afin=None):
    # calculate evolution and plot it
    if afin is None:
        afin = sp_0.r_isco()
    # no dm
    ev_0 =  inspiral.Classic.Evolve(sp_0, a0, e0, a_fin=afin,
                                    opt=inspiral.Classic.EvolutionOptions(accretion=False, verbose=verbose, accuracy=acc))
    plotEvolution(sp_0, ev_0, ax_a, ax_e, ax_ae=ax_ae, label=label + r'vacuum')

    ev_nacc = inspiral.Classic.Evolve(sp_dm, a0, e0, a_fin=afin,
                                    opt=inspiral.Classic.EvolutionOptions(accretion=False, haloPhaseSpaceDescription=True, verbose=verbose, accuracy=acc))
    l_nacc = plotEvolution(sp_dm, ev_nacc, ax_a, ax_e, ax_ae=ax_ae, label=label + r'no accretion')

    # with BH accretion
    m2=sp_dm.m2
    t_fin = ev_nacc.t[-1]
    ev_acc_BH = inspiral.Classic.Evolve(sp_dm, a0, e0, a_fin=afin, t_fin=t_fin,
                                    opt=inspiral.Classic.EvolutionOptions(accretion=True, accretionModel='Bondi-Hoyle',
                                                        haloPhaseSpaceDescription=True, verbose=verbose, accuracy=acc))
    l_acc_BH = plotEvolution(sp_dm, ev_acc_BH, ax_a, ax_e, ax_ae=ax_ae, ax_m=ax_m, label=label + r'Bondi-Hoyle, c=0', m2=m2)
    sp_dm.m2 = m2  # reverse accretion
    # with Classic accretion
    ev_acc = inspiral.Classic.Evolve(sp_dm, a0, e0, a_fin=afin,  t_fin=t_fin,
                                    opt=inspiral.Classic.EvolutionOptions(accretion=True, accretionModel='Bondi-Hoyle',
                                                        haloPhaseSpaceDescription=True, verbose=verbose, accuracy=acc, dm_soundspeed2=0.01))
    sp_dm.m2 = m2  # reverse accretion
    l_acc = plotEvolution(sp_dm, ev_acc, ax_a, ax_e, ax_ae=ax_ae, ax_m=ax_m, label=label + r'Bondi-Hoyle, c=0.01', m2=m2)
    # with Classic accretion
    ev_acc = inspiral.Classic.Evolve(sp_dm, a0, e0, a_fin=afin,  t_fin=t_fin,
                                    opt=inspiral.Classic.EvolutionOptions(accretion=True, accretionModel='Bondi-Hoyle',
                                                        haloPhaseSpaceDescription=True, verbose=verbose, accuracy=acc, dm_soundspeed2=0.5))
    sp_dm.m2 = m2  # reverse accretion
    l_acc = plotEvolution(sp_dm, ev_acc, ax_a, ax_e, ax_ae=ax_ae, ax_m=ax_m, label=label + r'Bondi-Hoyle, c=0.5', m2=m2)

plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#94a4a2", "#3f90da", "#ffa90e", "#bd1f01", "#832db6"])
fig_ae, (ax_a,ax_ae)  = plt.subplots(2, 1, figsize=(6,10))

a0 = 100 * sp_0.r_isco()
e0 = 1e-1

ax_m = ax_a.twinx()
compareAccretionModels(sp_0, sp_dm_dyn, a0, e0, ax_a=ax_a, ax_ae=ax_ae, ax_m=ax_m)


ax_a.grid(); ax_a.legend()
ax_ae.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

from imripy import halo, merger_system as ms, constants as c, inspiral, waveform, plot_utils as pu


m1 = 1e3 * c.solar_mass_to_pc
m2 = 10. * c.solar_mass_to_pc

sp_0 = ms.SystemProp(m1, m2, halo.ConstHalo(0.))

r_spike=0.54
rho_spike = 226*c.solar_mass_to_pc
alpha = 2.5
sp_dm_stat = ms.SystemProp(m1, m2, halo.Spike(rho_spike, r_spike, alpha))


p0 = 1e6 * sp_0.r_schwarzschild()
e0 = 0.6
a0 = p0/(1.-e0**2)
#afin = 1e3*sp_0.r_isco()
F0 = np.sqrt(sp_0.m_total()/a0**3)/ 2./np.pi
nOrbits = 10
t_end = nOrbits/ F0

ev_pp = inspiral.Classic.Evolve(sp_dm_stat, a0, e0, t_fin=t_end,
            opt=inspiral.Classic.EvolutionOptions(accretion=False, haloPhaseSpaceDescription=False, periapsePrecession=True, verbose=2))

fig_ae, (ax_a,ax_ae, ax_pa)  = plt.subplots(3, 1, figsize=(6,10))

pu.plotEvolution(sp_dm_stat, ev_pp, ax_a=ax_a, ax_ae=ax_ae, ax_pa=ax_pa)
#ax_pa.set_xscale('log')
ax_a.grid(); ax_ae.grid(); ax_pa.grid()

plt.show()


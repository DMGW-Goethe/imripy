import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from imripy import merger_system as ms, constants as c, inspiral, waveform, halo
import imripy.inspiral.forces as forces

m1 = 1e3 * c.solar_mass_to_pc
m2 = 1e1  * c.solar_mass_to_pc
D = 1e8
beta = 0.
iota = 0.

sp_0 = ms.SystemProp(m1, m2, halo.ConstHalo(0.), D, inclination_angle=iota, pericenter_angle=beta)

ln_Lambda=10.
rho_spike = 226 * c.solar_mass_to_pc
r_spike = 0.54
e0 = 0.6
a0 = 200./(1.-e0**2) * sp_0.m1
afin = 1. * sp_0.r_isco()

sp_1 = ms.SystemProp(m1, m2, halo.Spike(rho_spike, r_spike, 7./3.), D, inclination_angle=iota, pericenter_angle=beta)
sp_2 = ms.SystemProp(m1, m2, halo.Spike(rho_spike, r_spike, 2.),    D, inclination_angle=iota, pericenter_angle=beta)
sp_3 = ms.SystemProp(m1, m2, halo.Spike(rho_spike, r_spike, 1.5),   D, inclination_angle=iota, pericenter_angle=beta)

evOpt = inspiral.Classic.EvolutionOptions(dissipativeForces={forces.GWLoss(), forces.DynamicalFriction(ln_Lambda=ln_Lambda)}, accuracy=1e-10, verbose=1)

ev_0 = inspiral.Classic.Evolve_old(sp_0, a0, e0, a_fin=afin, opt=evOpt)
ev_1 = inspiral.Classic.Evolve_old(sp_1, a0, e0, a_fin=afin, opt=evOpt)
ev_2 = inspiral.Classic.Evolve_old(sp_2, a0, e0, a_fin=afin, opt=evOpt)
ev_3 = inspiral.Classic.Evolve_old(sp_3, a0, e0, a_fin=afin, opt=evOpt)

fig, (ax_p, ax_e) = plt.subplots(2, 1, figsize=(16,20))
ax_p.plot(ev_0.t/c.year_to_pc, ev_0.a*(1.-ev_0.e**2)/sp_0.m1, label='0')
ax_p.plot(ev_1.t/c.year_to_pc, ev_1.a*(1.-ev_1.e**2)/sp_1.m1, label='1')
ax_p.plot(ev_2.t/c.year_to_pc, ev_2.a*(1.-ev_2.e**2)/sp_2.m1, label='2')
ax_p.plot(ev_3.t/c.year_to_pc, ev_3.a*(1.-ev_3.e**2)/sp_3.m1, label='3')

ax_e.plot(ev_0.t/c.year_to_pc, ev_0.e, label='0')
ax_e.plot(ev_1.t/c.year_to_pc, ev_1.e, label='1')
ax_e.plot(ev_2.t/c.year_to_pc, ev_2.e, label='2')
ax_e.plot(ev_3.t/c.year_to_pc, ev_3.e, label='3')

ax_p.grid(); ax_p.set_xlabel('t / yr'); ax_p.set_ylabel(r'p / $GM/c^4$')
ax_p.legend();
ax_e.grid(); ax_e.set_xlabel('t / yr'); ax_e.set_ylabel('e')


def plotWaveform(sp, ev, ax_h, f_start, t_len=None, label=" "):
    F = np.sqrt(sp.m_total()/ev.a**3) /2./np.pi
    f_start = np.clip(f_start, F[0], F[-1])
    print("F=", F,"f_start=", f_start)
    t_start = interp1d(F, ev.t, kind='cubic', bounds_error=True)(f_start)
    print("t_start=", t_start/c.year_to_pc)
    if t_len is None:
        t_len = 10./f_start
    t_grid = np.linspace(t_start, t_start+t_len, int(20.*t_len*f_start))
    print(t_grid)
    h_plus, h_cross = waveform.h(sp, ev, t_grid)
    #ax_h.plot(t_grid, h_plus, label=r"$h_+^{" + label +"}$")
    ax_h.plot((t_grid-t_start)/c.s_to_pc, h_cross, label=r"$h_x^{" + label +"}$")

'''
fig, ax_h = plt.subplots(1,1, figsize=(20, 10))

plotWaveform(sp_0, ev_0, ax_h, f_start=1e-3*c.hz_to_invpc, label="0")
plotWaveform(sp_1, ev_1, ax_h, f_start=1e-3*c.hz_to_invpc, label="1")

ax_h.grid(); ax_h.legend();
'''
'''
a0 = 1e5/(1.-e0**2) * sp_0.m1
afin = 1. * sp_0.r_isco()
e0 = 0.6

t_0, a_0, e_0 = inspiral.Classic.Evolve(sp_0, a0, e0, afin)
t_1, a_1, e_1 = inspiral.Classic.Evolve(sp_1, a0, e0, afin, t_fin=t_0[-1]/1e6)
t_2, a_2, e_2 = inspiral.Classic.Evolve(sp_2, a0, e0, afin, t_fin=t_0[-1]/1e5)
t_3, a_3, e_3 = inspiral.Classic.Evolve(sp_3, a0, e0, afin, t_fin=t_0[-1]/1e4)

fig, (ax_p, ax_e) = plt.subplots(2, 1, figsize=(16,20))
ax_p.plot(t_0/c.year_to_pc, a_0*(1.-e_0**2)/sp_0.m1, label='0')
ax_p.plot(t_1/c.year_to_pc, a_1*(1.-e_1**2)/sp_1.m1, label='1')
ax_p.plot(t_2/c.year_to_pc, a_2*(1.-e_2**2)/sp_2.m1, label='2')
ax_p.plot(t_3/c.year_to_pc, a_3*(1.-e_3**2)/sp_3.m1, label='3')

ax_e.plot(t_0/c.year_to_pc, e_0, label='0')
ax_e.plot(t_1/c.year_to_pc, e_1, label='1')
ax_e.plot(t_2/c.year_to_pc, e_2, label='2')
ax_e.plot(t_3/c.year_to_pc, e_3, label='3')

ax_p.grid(); ax_p.set_xscale('log'); ax_p.set_xlabel('t / yr'); ax_p.set_ylabel(r'p / $GM/c^4$')
ax_p.legend();
ax_e.grid(); ax_e.set_xscale('log'); ax_e.set_xlabel('t / yr'); ax_e.set_ylabel('e')

'''
plt.show()


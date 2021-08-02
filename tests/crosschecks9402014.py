import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import imripy.merger_system as ms
import imripy.halo as halo
import imripy.inspiral as inspiral
import imripy.waveform as waveform
import imripy.detector as detector

solar_mass_in_pc = 4.8e-14
year_in_pc = 0.3064

#sp = ms.SystemProp(10**3 *solar_mass_in_pc, 1 * solar_mass_in_pc, 1e3, ms.SpikedNFW( 2.68e-13, 23.1, 0.54, 7./3.))
sp = ms.SystemProp(1e3 *solar_mass_in_pc, 1. * solar_mass_in_pc, halo.ConstHalo(0.))

print(sp.m_chirp(), sp.redshifted_m_chirp())

R0= 100.*sp.r_isco()
R_fin = sp.r_isco()
#R_fin =  [15.*sp.r_isco, sp.r_isco]

t, R = inspiral.Classic.evolve_circular_binary(sp, R0, R_fin, acc=1e-8)
omega_s = sp.omega_s(R)
#f_gw = np.geomspace(omega_s[0]/np.pi, omega_s[-1]/np.pi, num=200)
f_gw = omega_s/np.pi
omega_s_obs = omega_s/(1. + sp.z())
f_gw_obs = omega_s/np.pi
f_isco = f_gw[-1]
t_of_f = interp1d(omega_s/np.pi, t, kind='cubic', bounds_error=False, fill_value='extrapolate')
t_obs_of_f = interp1d(omega_s_obs/np.pi, t, kind='cubic', bounds_error=False, fill_value='extrapolate')
omega_gw = interp1d(t, 2*omega_s, kind='cubic', bounds_error=False, fill_value='extrapolate')
omega_gw_obs = interp1d(t, 2*omega_s_obs, kind='cubic', bounds_error=False, fill_value='extrapolate')

t_c = t[-1] - t[0] + 5./256. * R[-1]**4/sp.m_total()**2 / sp.m_reduced()
t_c_obs = t_c*(1.+sp.z())
t_c0 = 5./256. *R0**4 / sp.m_total()**2 / sp.m_reduced()
omega_isco = np.sqrt((sp.m1+sp.m2)/sp.r_isco()**3)
print(t_c0, t_c, t_c/t_c0 - 1.)
print(omega_isco, omega_s[-1], omega_s[-1]/omega_isco - 1.)
print(sp.r_isco(), R[-1], R[-1]/sp.r_isco() - 1.)

#plt.axvline(omega_gw(t_c0)/2./np.pi*year_in_pc*3.17e-8, label='$f_{isco}^{analytic}$')
plt.axvline(f_isco*year_in_pc*3.17e-8, label='$f^{obs}_{isco}$')

plt.plot(f_gw*year_in_pc*3.17e-8, t_obs_of_f(f_gw), label='$t_{obs}(f)$')
plt.plot(f_gw*year_in_pc*3.17e-8, (1.+sp.z())*(t[-1] - 5. * (8*np.pi*f_gw)**(-8./3.) * sp.m_chirp()**(-5./3.)), label='$t_{obs}(f)^{analytic}$')

#Phit = np.array([quad(lambda u: np.exp(u)*omega_gw(np.exp(u)), np.log(t[0]), np.log(y0))[0] for y0 in t ])
Phit = np.cumsum([quad(lambda t: omega_gw(t), t[i-1], t[i], limit=500, epsrel=1e-13, epsabs=1e-13)[0] if not i == 0 else 0. for i in range(len(t)) ])
#Phi = interp1d(t, Phit - Phit[-1], kind='cubic', bounds_error=False, fill_value='extrapolate')(t_of_f(f_gw))
Phi = Phit - Phit[-1]
Phi0 = - 2.*(8.*np.pi*sp.m_chirp()*f_gw)**(-5./3.) + 2.*(8.*np.pi*sp.m_chirp()*f_isco)**(-5./3.)

plt.plot(f_gw*year_in_pc*3.17e-8, Phi, label=r'$\Phi^{code}$')
plt.plot(f_gw*year_in_pc*3.17e-8, Phi0, label=r'$\Phi^{analytic}$')
plt.plot(f_gw*year_in_pc*3.17e-8, Phi - Phi0, label=r'$\Delta\Phi$')

#tpt = omega_gw(t_of_f(f_gw)) * (t_of_f(f_gw) -  t_of_f(f_isco))
tpt = 2.*np.pi*f_gw * (t - t_c)
tpt0 = -5./4. * (8.*np.pi*sp.m_chirp()*f_gw)**(-5./3.)

plt.plot(f_gw*year_in_pc*3.17e-8, tpt, label=r'$2\pi ft^{code}$')
plt.plot(f_gw*year_in_pc*3.17e-8, tpt0, label=r'$2\pi ft^{analytic}$')
plt.plot(f_gw*year_in_pc*3.17e-8, tpt - tpt0, label=r'$\Delta2\pi ft$')
plt.plot(f_gw*year_in_pc*3.17e-8, 2.*np.pi*f_gw*np.abs(t_c-t_c0), label=r'$\omega_{gw}\Delta t_c$')

PhiTild =  tpt - Phi
#PhiTild0 =  - 3./4. * (8.*np.pi*sp.m_chirp_redshifted*f_gw)**(-5./3.)
#PhiTild0 = PhiTild0 - PhiTild[-1]
PhiTild0 = tpt0 - Phi0

plt.plot(f_gw*year_in_pc*3.17e-8, PhiTild, label=r'$\tilde{\Phi}_0$')
plt.plot(f_gw*year_in_pc*3.17e-8, PhiTild0, label=r'$\tilde{\Phi}^{analytic}_0$')
plt.plot(f_gw*year_in_pc*3.17e-8, PhiTild - PhiTild0, label=r'$\Delta\tilde{\Phi}$')

plt.xscale('log');
plt.yscale('symlog')
plt.legend(); plt.grid()


plt.figure()

plt.axvline(t_c0/year_in_pc, label='$t_c^{analytic}$')

Ra = (256./5. * sp.m_reduced() * sp.m_total()**2 * (t_c - t))**(1./4.)
plt.plot(t/year_in_pc, R, label='$R^{code}$')
plt.plot(t/year_in_pc, Ra, label='$R^{analytic}$')
plt.plot(t/year_in_pc, np.abs(Ra - R), label='$\Delta R$')

Phi = Phit
Phi0 = -2.* (1./5.*(t_c - t)/ sp.m_chirp())**(5./8.)
plt.plot(t/year_in_pc, Phi, label='$\Phi^{code}$')
plt.plot(t/year_in_pc, Phi0  - Phi0[0] + Phi[0], label='$\Phi^{analytic}$')
plt.plot(t/year_in_pc, np.abs(Phi0 - Phi0[0] - Phi + Phi[0]), label='$\Delta\Phi$')

f_gw0 = 1./8./np.pi * 5**(3./8.) * sp.m_chirp()**(-5./8.) * (t_c-t)**(-3./8.) / (1.+sp.z())
plt.plot(t/year_in_pc, omega_s/np.pi, label='$f_{gw}$')
plt.plot(t/year_in_pc, f_gw0, label='$f_{gw}^{analytic}$')
plt.plot(t/year_in_pc, np.abs(omega_s/np.pi - f_gw0), label='$\Delta f_{gw}$' )
#plt.plot(t/year_in_pc, Phi, label='$\Phi(t)$')
#plt.plot(t/year_in_pc, omega_gw(t)*year_in_pc, label='$\omega_{gw}$')
#plt.plot(t/year_in_pc, omega_gw.derivative()(t) * year_in_pc, label='$\dot{\omega}_{gw}$')
#plt.plot(t/year_in_pc, A(t), label='A')
plt.xlabel('t / year');
#plt.xscale('log')
plt.yscale('log')
plt.legend(); plt.grid()


plt.figure()
f_gw0 = omega_s/np.pi
f_gw, h, _, Psi, __, PsiTild, __ = waveform.h_2(sp, t, omega_s, R, dbg=True)

Psi0 = 2.*np.pi*f_gw0 * (t_c0 + sp.D) - np.pi/4. + 3./4. * (8.*np.pi*sp.m_chirp()*f_gw0)**(-5./3.)
plt.plot(f_gw*year_in_pc*3.17e-8, Psi, label=r'$\Psi^{code}$')
plt.plot(f_gw0/(1.+sp.z())*year_in_pc*3.17e-8, Psi0, label=r'$\Psi^{analytic}$')

h0 = 1./sp.D * sp.redshifted_m_chirp()**(5./6.)*(f_gw0/(1.+sp.z()))**(-7./6.)
plt.plot(f_gw*year_in_pc*3.17e-8, h, label=r'$h^{code}$')
plt.plot(f_gw0/(1.+sp.z())*year_in_pc*3.17e-8, h0, label=r'$h^{analytic}$')

plt.plot(f_gw*year_in_pc*3.17e-8, PhiTild, label=r'$\tilde{\Phi}_0$')
plt.plot(f_gw0/(1.+sp.z())*year_in_pc*3.17e-8, PhiTild0, label=r'$\tilde{\Phi}^{analytic}_0$')

plt.xlabel('$f$')
plt.xscale('log')
plt.yscale('log')
plt.legend(); plt.grid()
'''
plt.figure()

htilde = interp1d(f_gw, h, kind='cubic', bounds_error=False, fill_value=(0.,0.))
SoN = detector.SignalToNoise(f_gw, htilde, detector.eLisa())

plt.plot(f_gw*year_in_pc*3.17e-8, SoN, label='$S/N$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$f$ / Hz')
plt.legend(); plt.grid()
'''
plt.show()


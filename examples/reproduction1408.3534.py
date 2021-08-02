import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from imripy import merger_system as ms
from imripy import inspiral
from imripy import halo
from imripy import waveform
from imripy import detector

m1 = 1e3 * ms.solar_mass_to_pc
m2 = 1 * ms.solar_mass_to_pc
D = 5e8

sp_0 = ms.SystemProp(m1, m2, halo.ConstHalo(0.), D)
sp_1 = ms.SystemProp(m1, m2, halo.SpikedNFW( 2.68e-13, 23.1, 0.54, 1.5), D)
sp_2 = ms.SystemProp(m1, m2, halo.SpikedNFW( 2.68e-13, 23.1, 0.54, 2.), D)
sp_3 = ms.SystemProp(m1, m2, halo.SpikedNFW( 2.68e-13, 23.1, 0.54, 7./3.), D)

def getObservablePhi(sp, R0, R_fin, acc=1e-14):
    t, R = inspiral.Classic.evolve_circular_binary(sp, R0, R_fin, t_0=0., acc=acc)
    omega_s = sp.omega_s(R)
    print(len(t), t)

    f_gw, h, _, Psi, t_of_f, PhiTild, A = waveform.h_2(sp, t, omega_s, R, dbg=True, acc=acc)

    return  f_gw, PhiTild, h

R0 = 50.*sp_1.r_isco()
#R_fin = [20.*sp_1.r_isco(), sp_1.r_isco()]
R_fin = sp_1.r_isco()

f_gw0, PhiT0, h0 = getObservablePhi(sp_0, R0, R_fin)
f_gw1, PhiT1, h1 = getObservablePhi(sp_1, R0, R_fin)
f_gw2, PhiT2, h2 = getObservablePhi(sp_2, R0, R_fin)
f_gw3, PhiT3, h3 = getObservablePhi(sp_3, R0, R_fin)

plt.figure()

PhiT0int = interp1d(f_gw0, PhiT0, kind='cubic', bounds_error=False, fill_value=(0., 0.))

Phi0 = - 2.*(8.*np.pi*sp_0.m_chirp()*f_gw0)**(-5./3.) + 2.*(8.*np.pi*sp_0.m_chirp()*f_gw0[-1])**(-5./3.)
t_c0 =  5./256. * R0**4/sp_0.m_total()**2 / sp_0.m_reduced()
tpt0 = -5./4. * (8.*np.pi*sp_0.m_chirp()*f_gw0)**(-5./3.)
PhiTild0 = tpt0 - Phi0
#PhiT0int = interp1d(f_gw0,  PhiTild0, kind='cubic', bounds_error=False, fill_value=(0., 0.))

plt.plot(f_gw0/ms.hz_to_invpc , PhiT0 , label=r'$\tilde{\Phi}_0$')
plt.plot(f_gw0/ms.hz_to_invpc,  PhiTild0, label=r'$\tilde{\Phi}^{analytic}_0$')
plt.plot(f_gw0/ms.hz_to_invpc, np.abs(PhiT0 - PhiTild0) , label=r'$\Delta\tilde{\Phi}_0$')

plt.plot(f_gw1/ms.hz_to_invpc, np.abs(PhiT1 - PhiT0int(f_gw1)) , label='$\Delta\Phi_1$')
plt.plot(f_gw2/ms.hz_to_invpc, np.abs(PhiT2 - PhiT0int(f_gw2)) , label='$\Delta\Phi_2$')
plt.plot(f_gw3/ms.hz_to_invpc, np.abs(PhiT3 - PhiT0int(f_gw3)) , label='$\Delta\Phi_3$')

plt.xlabel('f / Hz'); plt.xscale('log')
plt.yscale('log')
plt.legend(); plt.grid()

plt.figure()

h0int = interp1d(f_gw0, h0, kind='cubic', bounds_error=False, fill_value=(0.,0.))

plt.plot(f_gw0/ms.hz_to_invpc, h0 , label='$h_0$')

plt.plot(f_gw1/ms.hz_to_invpc, h1 , label='$h_1$')
plt.plot(f_gw1/ms.hz_to_invpc, np.abs(h1-h0int(f_gw1)) , label='$\Delta h_1$')

plt.plot(f_gw2/ms.hz_to_invpc, h2 , label='$h_2$')
plt.plot(f_gw2/ms.hz_to_invpc, np.abs(h2-h0int(f_gw2)) , label='$\Delta h_2$')

plt.plot(f_gw3/ms.hz_to_invpc, h3 , label='$h_3$')
plt.plot(f_gw3/ms.hz_to_invpc, np.abs(h3-h0int(f_gw3)) , label='$\Delta h_3$')

plt.xscale('log')
plt.xlabel('f')
plt.yscale('log')
plt.ylabel(r'$|\tilde{h}|$')
plt.legend(); plt.grid()

plt.figure()

plt.plot(f_gw0/ms.hz_to_invpc, 2.*f_gw0*h0 , label='$h_c^0$')

plt.plot(f_gw1/ms.hz_to_invpc, 2.*f_gw1*h1 , label='$h_c^1$')

plt.plot(f_gw2/ms.hz_to_invpc, 2.*f_gw2*h2 , label='$h_c^2$')

plt.plot(f_gw3/ms.hz_to_invpc, 2.*f_gw3*h3 , label='$h_c^3$')

f = np.geomspace(detector.Lisa().Bandwith()[0], detector.Lisa().Bandwith()[1], 100)
plt.plot(f/ms.hz_to_invpc, detector.Lisa().NoiseStrain(f), label='LISA')

plt.xscale('log')
plt.xlabel('f')
plt.yscale('log')
plt.ylabel('characteristic strain')
plt.legend(); plt.grid()

plt.figure()

SoN0 = detector.SignalToNoise(f_gw0, interp1d(f_gw0, h0, kind='cubic', bounds_error=False, fill_value=(0.,0.)),  detector.Lisa())
SoN1 = detector.SignalToNoise(f_gw1, interp1d(f_gw1, h1, kind='cubic', bounds_error=False, fill_value=(0.,0.)),  detector.Lisa())
SoN2 = detector.SignalToNoise(f_gw2, interp1d(f_gw2, h2, kind='cubic', bounds_error=False, fill_value=(0.,0.)),  detector.Lisa())
SoN3 = detector.SignalToNoise(f_gw3, interp1d(f_gw3, h3, kind='cubic', bounds_error=False, fill_value=(0.,0.)),  detector.Lisa())

plt.plot(f_gw0/ms.hz_to_invpc, SoN0 , label='$S/N_0$')
plt.plot(f_gw1/ms.hz_to_invpc, SoN1 , label='$S/N_1$')
plt.plot(f_gw2/ms.hz_to_invpc, SoN2 , label='$S/N_2$')
plt.plot(f_gw3/ms.hz_to_invpc, SoN3 , label='$S/N_3$')

plt.xscale('log')
plt.xlabel('f / Hz')
plt.yscale('log')
plt.ylabel('S/N')
plt.legend(); plt.grid();

plt.show()


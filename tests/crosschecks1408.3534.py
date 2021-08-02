import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt

from imripy import merger_system as ms
from imripy import inspiral
from imripy import waveform
from imripy import halo

solar_mass_in_pc = 4.8e-14
year_in_pc = 0.3064

inspiral.Classic.ln_Lambda=3.

def Meff(sp, r):
    return np.where(r > sp.r_isco(), sp.m1 - 4.*np.pi*sp.halo.rho_spike*sp.halo.r_spike**3 *sp.r_isco()**(3.-sp.halo.alpha) /(3.-sp.halo.alpha), sp.m1)

def F(sp, r):
    return np.where(r > sp.r_isco(), 4.*np.pi * sp.halo.rho_spike*sp.halo.r_spike**sp.halo.alpha /(3.-sp.halo.alpha), 0.)

def plotHalo(sp, r0, r1):
    r = np.geomspace(r0, r1, num=100)
    plt.loglog(r/sp.r_isco(), sp.halo.density(r), label=r'$\rho_{dm}$')
    plt.loglog(r/sp.r_isco(), sp.halo.mass(r), label='$m_{dm}$')
    plt.loglog(r/sp.r_isco(), sp.mass(r), label='$m_{tot}$')
    plt.loglog(r/sp.r_isco(), Meff(sp,r) + F(sp,r)*r**(3.-sp.halo.alpha), label=r'$M_{eff}+r^{3-\alpha} F$')
    plt.loglog(r/sp.r_isco(), F(sp,r)*r**(3.-sp.halo.alpha), label=r'$r^{3-\alpha} F$')
    #spl = US(np.log(r), 4*np.pi*r**3 * sp.halo.density(r), k=3)
    #plt.loglog(r/sp.r_isco, spl.antiderivative()(np.log(r)), label=r'$\int \rho_{dm}$')
    plt.xlabel('$r/r_{ISCO}$')

def plotOmega_s(sp, r0, r1):
    r = np.geomspace(r0, r1, num=100)
    plt.loglog(r/sp.r_isco(), sp.omega_s(r), label=r'$\omega_s^{code}$')
    plt.loglog(r/sp.r_isco(), np.sqrt(Meff(sp, r)/r**3 + F(sp, r)/r**(sp.halo.alpha)),label=r'$\omega_s^{paper}$' )
    plt.xlabel('$r/r_{ISCO}$')


def coeffs(sp):
    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp, 2.*sp.r_isco())
    c_gw = 256./5.* sp.m2 * Meff(sp, 2.*sp.r_isco())**2 * eps**(4./(3.-alpha))
    c_df = 8.*np.pi*sp.m2 *sp.halo.rho_spike *sp.halo.r_spike**alpha * 3. \
            * Meff(sp, 2.*sp.r_isco())**(-3./2.)* eps**((2.*alpha-3.)/(6.-2.*alpha))
    ctild = c_df/c_gw
    return c_gw, c_df, ctild


def f_gw(x, alpha):
    return (1.+x**(3.-alpha))**3 / ( 4.*x**3 * ( 1.+ (4.-alpha) *x**(3.-alpha) ) )

def f_df(x, alpha):
    return 1. / ( (1.+x**(3.-alpha))**(1./2.) * ( 1.+ (4.-alpha) *x**(3.-alpha) )* x**(-5./2.+alpha) )

def plotDiffEq(sp, r0, r1):
    r = np.geomspace(r0, r1, num=100)
    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp, 2.*sp.r_isco())
    x = eps**(1./(3.-alpha))*r
    c_gw, c_df, ctild = coeffs(sp)
    print(c_gw*year_in_pc, c_df*year_in_pc)
    l, = plt.loglog(r/sp.r_isco(), np.abs(inspiral.Classic.dE_gw_dt(sp, r))/inspiral.Classic.dE_orbit_da(sp, r), label=r'$dE_{gw}/dt / dE_{orbit}/dR$', alpha =0.5)
    plt.loglog(r/sp.r_isco(), c_gw*f_gw(x, alpha) , label='$c_{gw}f_{gw}$', color=l.get_c(), linestyle='--')
    l, = plt.loglog(r/sp.r_isco(), np.abs(inspiral.Classic.dE_df_dt(sp, r))/inspiral.Classic.dE_orbit_da(sp, r), label=r'$dE_{df}/dt / dE_{orbit}/dR$', alpha = 0.5)
    plt.loglog(r/sp.r_isco(), c_df* f_df(x, alpha), label='$c_{df}f_{df}$', color=l.get_c(), linestyle='--')
    plt.xlabel('$r/r_{ISCO}$')

def J(x, alpha):
    return 4. * x**(11./2. - alpha) / (1. + x**(3.-alpha))**(7./2.)

def K(x, alpha):
    return (1.+x**(3.-alpha))**(5./2.) * (1. + alpha/3.*x**(3.-alpha)) / (1. + (4.-alpha)*x**(3-alpha) )

def plotPhiprimeprime(sp, r0, r1):
    r = np.geomspace(r0, r1, num=100)
    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp, 2.*sp.r_isco())
    x = eps**(1./(3.-alpha))*r
    c_gw, c_df, ctild = coeffs(sp)

    plt.loglog(r/sp.r_isco(), Meff(sp, 2.*sp.r_isco())**(1./2.) * eps**(3./2./(3.-alpha)) \
                    * c_gw*(1.+ctild*J(x, alpha)) *3./4.* K(x,alpha) * x**(-11./2.), label=r'$\ddot{\Phi}^{paper}$' )
    #plt.loglog(r/sp.r_isco(), Meff(sp, 2.*sp.r_isco)**(1./2.) * eps**(3./2./(3.-alpha)) \
    #                * (c_gw*f_gw(x, alpha) + c_df*f_df(x, alpha)) * (3. + alpha*x**(3.-alpha))/(x**(5./2.) * (1.+ x**(3.-alpha))**(1./2.) ), label=r'$\ddot{\Phi}^{paper,ref}$' )
    plt.loglog(r/sp.r_isco(), (sp.mass(r)/r**3 )**(-1./2.) * (-3.*sp.mass(r)/r**4 + 4.*np.pi *sp.halo.density(r)/r )* inspiral.Classic.da_dt(sp, r), label=r'$\ddot{\Phi}^{code}$')
    plt.xlabel(r'$r/r_{ISCO}$')


def L(sp, f):
    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp, 2.*sp.r_isco())
    c_gw, c_df, ctild = coeffs(sp)
    c_eps = Meff(sp, 2.*sp.r_isco())**(11./6.-1./3.*alpha) * ctild * eps**((11.-2.*alpha)/(6.-2.*alpha))
    deltatild = (1./np.pi**2 / f**2)**(1.-alpha/3.)
    return 1. + 4.*c_eps*deltatild**((11.-2.*alpha)/(6.-2.*alpha))


def phaseIntegrand(sp, f):
    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp, 2.*sp.r_isco())
    delta = (Meff(sp, 2.*sp.r_isco())/ np.pi**2 / f**2)**((3.-alpha)/3.)
    chi = 1. + delta*eps/3. + (2.-alpha)/9. *delta**2 * eps**2
    x = (delta*eps)**(1./(3.-alpha)) *chi
    c_gw, c_df, ctild = coeffs(sp)
    return chi**(11./2.) / f**(8./3.) / K(x, alpha) / (1. + ctild*J(x,alpha))

def plotPhase(sp, t, R, omega_s):
    #f = np.geomspace(omega_s[1], omega_s[-2], num=200)/np.pi
    #f_isco = f[-1]
    f_gw = omega_s/np.pi
    f_isco = f_gw[-1]
    t_c = t[-1] + 5./256. * R[-1]**4/sp.m_total()**2 / sp.m_reduced()

    PhiTild0 =  - 3./4. * (8.*np.pi*sp.m_chirp()*f_gw)**(-5./3.) + 3./4.*(8.*np.pi*sp.m_chirp()*f_isco)**(-5./3.)
    #plt.plot(f_gw*year_in_pc*3.17e-8, PhiTild0, label=r'$\tilde{\Phi}_0^{analytic}$')

    t_of_f = interp1d(omega_s/np.pi, t, kind='cubic', bounds_error=True)
    #omega_gw = UnivariateSpline(t, 2*omega_s, ext=1, k=5 )
    omega_gw = interp1d(t, 2*omega_s, kind='cubic', bounds_error=False, fill_value='extrapolate' )
    #Phit = np.array([quad(lambda u: np.exp(u)*omega_gw(np.exp(u)), np.log(t[0]), np.log(y0))[0] for y0 in t ])
    Phit = np.cumsum([quad(lambda t: omega_gw(t), t[i-1], t[i], limit=500, epsrel=1e-13, epsabs=1e-13)[0] if not i == 0 else 0. for i in range(len(t)) ])
    #Phi = interp1d(t, Phit - Phit[-1], kind='cubic', bounds_error=False, fill_value='extrapolate')(t_of_f(f_gw))
    Phi = Phit - Phit[-1]

    tpt = 2.*np.pi*f_gw * (t - t[-1])

    PhiTild = tpt - Phi
    DeltaPhi = PhiTild - PhiTild0
    plt.plot(f_gw*year_in_pc*3.17e-8, Phi, label=r'$\Phi^{code}$')
    plt.plot(f_gw*year_in_pc*3.17e-8, tpt, label=r'$2\pi t^{code}$')
    plt.plot(f_gw*year_in_pc*3.17e-8, PhiTild, label=r'$\tilde{\Phi}^{code}$')
    plt.plot(f_gw*year_in_pc*3.17e-8, DeltaPhi, label=r'$\Delta\tilde{\Phi}^{code}$')

    #integrand = UnivariateSpline(np.log(f), f* f**(-8./3.)/L(sp, f), k=5)
    #integrand = UnivariateSpline(np.log(f), f* phaseIntegrand(sp, f), k=5)
    #Phi = integrand.antiderivative()
    #Phi = np.array([quad(lambda f: np.exp(-5./3.*f)/L(sp, np.exp(f)), np.log(f_gw[0]), np.log(y0))[0] for y0 in f_gw ])
    Phi = np.cumsum([quad(lambda f: f**(-8./3.)/L(sp, f), f_gw[i-1], f_gw[i], limit=200, epsrel=1e-13, epsabs=1e-13)[0] if not i == 0 else 0. for i in range(len(f_gw)) ])
    Phi = 10./3. * (8.*np.pi*sp.m_chirp())**(-5./3.) * (Phi - Phi[-1])
    #integrand2 = UnivariateSpline(np.log(f), f * f**(-11./3.)/L(sp, f), k=5)
    #integrand2 = UnivariateSpline(np.log(f), phaseIntegrand(sp, f), k=5)
    #tpt = integrand2.antiderivative()
    #tpt = np.array([quad(lambda f: np.exp(-8./3.*f)/L(sp, np.exp(f)), np.log(f_gw[0]), np.log(y0))[0] for y0 in f_gw ])
    tpt = np.cumsum([quad(lambda f: f**(-11./3.)/L(sp, f), f_gw[i-1], f_gw[i], limit=200, epsrel=1e-13, epsabs=1e-13)[0] if not i==0 else 0. for i in range(len(f_gw)) ])
    tpt = 10./3. * (8.*np.pi*sp.m_chirp())**(-5./3.) * f_gw * ( tpt - tpt[-1])
    PhiTild = tpt - Phi
    DeltaPhi = PhiTild - PhiTild0

    plt.plot(f_gw*year_in_pc*3.17e-8, Phi, label=r'$\Phi^{paper}$')
    plt.plot(f_gw*year_in_pc*3.17e-8, tpt, label=r'$2\pi t^{paper}$')
    plt.plot(f_gw*year_in_pc*3.17e-8, PhiTild, label=r'$\tilde{\Phi}^{paper}$')
    plt.plot(f_gw*year_in_pc*3.17e-8, DeltaPhi, label=r'$\Delta\tilde{\Phi}^{paper}$')
    plt.xlabel('f')
    plt.xscale('log')
    plt.yscale('symlog')


def plotWaveform(sp, t, R, omega_s):
    #f = np.geomspace(omega_s[1], omega_s[-2], num=500)/np.pi
    f_gw, h, _, Psi  = waveform.h_2( sp, t, omega_s, R)
    plt.loglog(f_gw*year_in_pc*3.17e-8, h, label=r'$\tilde{h}^{code}$')

    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp, 2.*sp.r_isco())
    A = (5./24.)**(1./2.) * np.pi**(-2./3.) /sp.D * sp.m_chirp()**(5./6.)
    plt.loglog(f_gw*year_in_pc*3.17e-8, A*f_gw**(-7./6.) * (L(sp,f_gw))**(-1./2.), label=r'$\tilde{h}^{paper,approx}$')

    delta = (Meff(sp, 2.*sp.r_isco())/np.pi**2 / f_gw**2)**(1.-alpha/3.)
    chi = 1. + delta*eps/3. + (2.-alpha)/9. *delta**2 * eps**2
    x = (delta*eps)**(1./(3.-alpha)) *chi
    c_gw, c_df, ctild = coeffs(sp)
    plt.loglog(f_gw*year_in_pc*3.17e-8, A*f_gw**(-7./6.) * chi**(19./4.) * (K(x, alpha)* (1. + ctild*J(x, alpha)))**(-1./2.), label=r'$\tilde{h}^{paper}$' )
    plt.ylabel('h'); plt.xlabel('f')


m1 = 10**3 *solar_mass_in_pc
m2 = 1 *solar_mass_in_pc
D = 1e3
#sp_0 = ms.SystemProp(m1, m2, 1e3, ms.ConstHalo(0))
sp_1 = ms.SystemProp(m1, m2, halo.SpikedNFW( 2.68e-13, 23.1, 0.54, 7./3.), D)

plotHalo(sp_1,4./6.*sp_1.r_isco(), 1e4)
plt.legend(); plt.grid()

plt.figure()
plotOmega_s(sp_1, 4./6.*sp_1.r_isco(), 1e4)
plt.legend(); plt.grid()

plt.figure()
plotDiffEq(sp_1, sp_1.r_isco(), 1e7*sp_1.r_isco())
plt.legend(); plt.grid()

plt.figure()
plotPhiprimeprime(sp_1, sp_1.r_isco(), 1e5*sp_1.r_isco())
plt.legend(); plt.grid()

R0 = 80.*sp_1.r_isco()
t, R = inspiral.Classic.evolve_circular_binary(sp_1, R0, sp_1.r_isco(), acc=1e-11)
omega_s = sp_1.omega_s(R)

plt.figure()
plotPhase(sp_1, t, R, omega_s)
plt.legend(); plt.grid()

plt.figure()
plotWaveform(sp_1, t, R, omega_s)
plt.legend(); plt.grid()


plt.show()

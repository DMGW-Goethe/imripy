import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad
import collections
from imripy import halo
from imripy import merger_system as ms
from imripy import inspiral
from imripy import waveform


inspiral.Classic.ln_Lambda=3.

def Meff(sp, r):
    return np.where(r > sp.r_isco(), sp.m1 - 4.*np.pi*sp.halo.rho_spike*sp.halo.r_spike**3 *sp.r_isco()**(3.-sp.halo.alpha) /(3.-sp.halo.alpha), sp.m1)

def F(sp, r):
    return np.where(r > sp.r_isco(), 4.*np.pi * sp.halo.rho_spike*sp.halo.r_spike**sp.halo.alpha /(3.-sp.halo.alpha), 0.)

def coeffs(sp):
    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp, 2.*sp.r_isco())
    m2 = sp.m2
    if isinstance(m2, (np.ndarray, collections.Sequence)):
        m2 = m2[-1]
    c_gw = 256./5.* m2 * Meff(sp, 2.*sp.r_isco())**2 * eps**(4./(3.-alpha))
    c_df = 8.*np.pi*m2 *sp.halo.rho_spike *sp.halo.r_spike**alpha * 3. \
            * Meff(sp, 2.*sp.r_isco())**(-3./2.)* eps**((2.*alpha-3.)/(6.-2.*alpha))
    ctild = c_df/c_gw
    return c_gw, c_df, ctild

def b_A(sp, x, alpha):
    eps = F(sp,2.*sp.r_isco())/Meff(sp, 2.*sp.r_isco())
    r = x/eps**(1./(3.-alpha))
    omega_s = np.sqrt(Meff(sp, r)/r**3 + F(sp, r)/r**(sp.halo.alpha))
    return 4. * r**2 * omega_s**2 / inspiral.Classic.ln_Lambda  * (1. + r**2 * omega_s**2)

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
    print(c_gw*ms.year_to_pc, c_df*ms.year_to_pc)
    l, = plt.loglog(r/sp.r_isco(), np.abs(inspiral.Classic.dE_gw_dt(sp, r))/inspiral.Classic.dE_orbit_da(sp, r), label=r'$dE_{gw}/dt / dE_{orbit}/dR$', alpha=0.5)
    plt.loglog(r/sp.r_isco(), c_gw*f_gw(x, alpha) , label='$c_{gw}f_{gw}$', color=l.get_c(), linestyle='--')
    l, = plt.loglog(r/sp.r_isco(), np.abs(inspiral.Classic.dE_df_dt(sp, r))/inspiral.Classic.dE_orbit_da(sp, r), label=r'$dE_{df}/dt / dE_{orbit}/dR$', alpha=0.5)
    plt.loglog(r/sp.r_isco(), c_df* f_df(x, alpha), label='$c_{df}f_{df}$' , color=l.get_c(), linestyle='--')
    l, = plt.loglog(r/sp.r_isco(), np.abs(inspiral.Classic.dE_acc_dt(sp, r))/inspiral.Classic.dE_orbit_da(sp, r), label=r'$dE_{acc}/dt / dE_{orbit}/dR$', alpha=0.5)
    plt.loglog(r/sp.r_isco(), c_df* f_df(x, alpha)*b_A(sp, x, alpha), label='$c_{df}f_{df}b_A$' , color=l.get_c(), linestyle='--')
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
                    * c_gw*(1.+ctild*J(x, alpha)*(1.+b_A(sp, x, alpha))) *3./4.* K(x,alpha) * x**(-11./2.), label=r'$\ddot{\Phi}^{paper}$' )
    #plt.loglog(r/sp.r_isco(), Meff(sp, 2.*sp.r_isco)**(1./2.) * eps**(3./2./(3.-alpha)) \
    #                * (c_gw*f_gw(x, alpha) + c_df*f_df(x, alpha)) * (3. + alpha*x**(3.-alpha))/(x**(5./2.) * (1.+ x**(3.-alpha))**(1./2.) ), label=r'$\ddot{\Phi}^{paper,ref}$' )
    plt.loglog(r/sp.r_isco(), (sp.mass(r)/r**3 )**(-1./2.) * (-3.*sp.mass(r)/r**4 + 4.*np.pi *sp.halo.density(r)/r )* inspiral.Classic.da_dt(sp, r), label=r'$\ddot{\Phi}^{code}$')
    plt.xlabel(r'$r/r_{ISCO}$')


def L(sp, f):
    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp, 2.*sp.r_isco())
    c_gw, c_df, ctild = coeffs(sp)
    c_eps = Meff(sp, 2.*sp.r_isco())**(11./6.-1./3.*alpha) * ctild * eps**((11.-2.*alpha)/(6.-2.*alpha))
    b_eps = (np.pi*f * Meff(sp, 2.*sp.r_isco()))**(2./3.) / inspiral.Classic.ln_Lambda * (1. + (np.pi*f * Meff(sp, 2.*sp.r_isco()))**(2./3.))
    deltatild = (1./np.pi**2 / f**2)**(1.-alpha/3.)
    return 1. + 4.*c_eps*deltatild**((11.-2.*alpha)/(6.-2.*alpha)) * (1. + b_eps)


def phaseIntegrand(sp, f):
    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp, 2.*sp.r_isco())
    delta = (Meff(sp, 2.*sp.r_isco())/ np.pi**2 / f**2)**((3.-alpha)/3.)
    chi = 1. + delta*eps/3. + (2.-alpha)/9. *delta**2 * eps**2
    x = (delta*eps)**(1./(3.-alpha)) *chi
    c_gw, c_df, ctild = coeffs(sp)
    return chi**(11./2.) / f**(8./3.) / K(x, alpha) / (1. + ctild*J(x,alpha)*(1. + b_A(sp, x, alpha)))

def plotPhase(sp, t, R, omega_s):
    #f = np.geomspace(omega_s[1], omega_s[-2], num=200)/np.pi
    #f_isco = f[-1]
    f_gw = omega_s/np.pi
    f_isco = f_gw[-1]
    t_c = t[-1] + 5./256. * R[-1]**4/sp.m_total()**2 / sp.m_reduced()
    if isinstance(t_c, (np.ndarray, collections.Sequence)):
        t_c = t_c[-1]

    PhiTild0 =  - 3./4. * (8.*np.pi*sp.m_chirp()*f_gw)**(-5./3.) + 3./4.*(8.*np.pi*sp.m_chirp()*f_isco)**(-5./3.)
    #plt.plot(f_gw*ms.year_to_pc*3.17e-8, PhiTild0, label=r'$\tilde{\Phi}_0^{analytic}$')

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
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, Phi, label=r'$\Phi^{code}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, tpt, label=r'$2\pi t^{code}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, PhiTild, label=r'$\tilde{\Phi}^{code}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, DeltaPhi, label=r'$\Delta\tilde{\Phi}^{code}$')

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

    plt.plot(f_gw*ms.year_to_pc*3.17e-8, Phi, label=r'$\Phi^{paper}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, tpt, label=r'$2\pi t^{paper}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, PhiTild, label=r'$\tilde{\Phi}^{paper}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, DeltaPhi, label=r'$\Delta\tilde{\Phi}^{paper}$')
    plt.xlabel('f')
    plt.xscale('log')
    plt.yscale('symlog')


def plotWaveform(sp, t, R, omega_s):
    #f = np.geomspace(omega_s[1], omega_s[-2], num=500)/np.pi
    f_gw, h, _, Psi  = waveform.h_2( sp, t, omega_s, R)
    plt.loglog(f_gw*ms.year_to_pc*3.17e-8, h, label=r'$\tilde{h}^{code}$')

    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp, 2.*sp.r_isco())
    A = (5./24.)**(1./2.) * np.pi**(-2./3.) /sp.D * sp.m_chirp()**(5./6.)
    plt.loglog(f_gw*ms.year_to_pc*3.17e-8, A*f_gw**(-7./6.) * (L(sp,f_gw))**(-1./2.), label=r'$\tilde{h}^{paper,approx}$')

    delta = (Meff(sp, 2.*sp.r_isco())/np.pi**2 / f_gw**2)**(1.-alpha/3.)
    chi = 1. + delta*eps/3. + (2.-alpha)/9. *delta**2 * eps**2
    x = (delta*eps)**(1./(3.-alpha)) *chi
    c_gw, c_df, ctild = coeffs(sp)
    plt.loglog(f_gw*ms.year_to_pc*3.17e-8, A*f_gw**(-7./6.) * chi**(19./4.) * (K(x, alpha)* (1. + ctild*J(x, alpha)*(1.+b_A(sp, x, alpha)) ))**(-1./2.), label=r'$\tilde{h}^{paper}$' )
    plt.ylabel('h'); plt.xlabel('f')


m1 = 1e3 *ms.solar_mass_to_pc
m2 = 1. *ms.solar_mass_to_pc
D = 1e3
#sp_0 = ms.SystemProp(m1, m2, 1e3, ms.ConstHalo(0))
sp_1 = ms.SystemProp(m1, m2, halo.SpikedNFW( 2.68e-13, 23.1, 0.54, 7./3.), D)


plt.figure()
plotDiffEq(sp_1, sp_1.r_isco(), 1e7*sp_1.r_isco())
plt.legend(); plt.grid()

plt.figure()
plotPhiprimeprime(sp_1, sp_1.r_isco(), 1e5*sp_1.r_isco())
plt.legend(); plt.grid()

R0 = 100.*sp_1.r_isco()
t, R, m2 = inspiral.Classic.evolve_circular_binary(sp_1, R0, sp_1.r_isco(), acc=1e-11, accretion=True)

sp_1.m2=m2
omega_s = sp_1.omega_s(R)

plt.figure()
plotPhase(sp_1, t, R, omega_s)
plt.legend(); plt.grid()

plt.figure()
plotWaveform(sp_1, t, R, omega_s)
plt.legend(); plt.grid()

plt.figure()
plt.loglog(t, m2/ms.solar_mass_to_pc, label="$m_2$")
plt.legend(); plt.grid()
print("mass increase:", m2[-1]/m2[0] -1.)


# Now check the eccentric implementation with a tiny eccentricity, it should be very similar
a0 = 100.*sp_1.r_isco()
e0 = 0.001
sp_1.m2 = 1.*ms.solar_mass_to_pc

t2, a2, e2, m22 = inspiral.Classic.evolve_elliptic_binary(sp_1, a0, e0, sp_1.r_isco(), acc=1e-11, accretion=True)

plt.figure()
plt.loglog(t, R, label='R, cirlular')
plt.loglog(t2, a2, label='a, elliptic')

plt.loglog(t, m2, label='$m_2$, cirlular')
plt.loglog(t2, m22, label='$m_2$, elliptic')

plt.loglog(t2, e2, label='e')
plt.grid(); plt.legend()


plt.show()



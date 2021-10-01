import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad, odeint, solve_ivp
import collections
from imripy import halo
from imripy import merger_system as ms
from imripy import inspiral
from imripy import waveform


inspiral.Classic.ln_Lambda=3.
inspiral.Classic.dmPhaseSpaceFraction=1.

def Meff(sp, r=None):
    """
    Returns Meff as given by eq (4), default is for r > r_min=r_isco
    """
    if r is None:
        r = 2.*sp.r_isco()
    return np.where(r > sp.r_isco(), sp.m1 - 4.*np.pi*sp.halo.rho_spike*sp.halo.r_spike**3 *sp.r_isco()**(3.-sp.halo.alpha) /(3.-sp.halo.alpha), sp.m1)

def F(sp, r=None):
    """
    Returns F as given by eq (5), default is for r > r_min=r_isco
    """
    if r is None:
        r = 2.*sp.r_isco()
    return np.where(r > sp.r_isco(), 4.*np.pi * sp.halo.rho_spike*sp.halo.r_spike**sp.halo.alpha /(3.-sp.halo.alpha), 0.)

def coeffs(sp):
    r"""
    Calculates the coefficients c_gw, c_df, and \tilde{c} as given by eq (18),(19), and (21)
    """
    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp)
    m2 = sp.m2
    if isinstance(m2, (np.ndarray, collections.Sequence)):
        m2 = m2[-1]
    c_gw = 256./5.* m2 * Meff(sp)**2 * eps**(4./(3.-alpha))
    c_df = 8.*np.pi*m2 *sp.halo.rho_spike *sp.halo.r_spike**alpha * 3. \
            * Meff(sp)**(-3./2.)* eps**((2.*alpha-3.)/(6.-2.*alpha))
    ctild = c_df/c_gw
    return c_gw, c_df, ctild

def b_A(sp, x, alpha):
    """
    Calculates b_A as given by equation (14), but as a function of x as in eq (15)
    """
    eps = F(sp)/Meff(sp)
    r = x/eps**(1./(3.-alpha))
    omega_s = np.sqrt(Meff(sp, r)/r**3 + F(sp, r)/r**(sp.halo.alpha))
    return 4. * r**2 * omega_s**2 / inspiral.Classic.ln_Lambda  * (1. + r**2 * omega_s**2)

def f_gw(x, alpha):
    """
    Calculates f_gw as given by eq (17) and (20)
    """
    return (1.+x**(3.-alpha))**3 / ( 4.*x**3 * ( 1.+ (4.-alpha) *x**(3.-alpha) ) )

def f_df(x, alpha):
    """
    Calculates f_dx as given by eq (17) and (20)
    """
    return 1. / ( (1.+x**(3.-alpha))**(1./2.) * ( 1.+ (4.-alpha) *x**(3.-alpha) )* x**(-5./2.+alpha) )

def plotDiffEq(sp, r0, r1):
    """
    This function plots the differential equation parts of eq(20) and plots them vs the counterpart in the numerical code in between r0 and r1
    """
    r = np.geomspace(r0, r1, num=100)
    alpha = sp.halo.alpha
    eps = F(sp)/Meff(sp)
    x = eps**(1./(3.-alpha))*r
    c_gw, c_df, ctild = coeffs(sp)
    print(c_gw*ms.year_to_pc, c_df*ms.year_to_pc)
    l, = plt.loglog(r/sp.r_isco(), np.abs(inspiral.Classic.dE_gw_dt(sp, r))/inspiral.Classic.dE_orbit_da(sp, r), label=r'$dE_{gw}/dt / dE_{orbit}/dR$', alpha=0.5)
    plt.loglog(r/sp.r_isco(), c_gw*f_gw(x, alpha)/eps**(1./(3.-alpha)) , label='$c_{gw}f_{gw}$', color=l.get_c(), linestyle='--')
    l, = plt.loglog(r/sp.r_isco(), np.abs(inspiral.Classic.dE_df_dt(sp, r))/inspiral.Classic.dE_orbit_da(sp, r), label=r'$dE_{df}/dt / dE_{orbit}/dR$', alpha=0.5)
    plt.loglog(r/sp.r_isco(), c_df* f_df(x, alpha)/eps**(1./(3.-alpha)), label='$c_{df}f_{df}$' , color=l.get_c(), linestyle='--')
    l, = plt.loglog(r/sp.r_isco(), np.abs(inspiral.Classic.dE_acc_dt(sp, r))/inspiral.Classic.dE_orbit_da(sp, r), label=r'$dE_{acc}/dt / dE_{orbit}/dR$', alpha=0.5)
    plt.loglog(r/sp.r_isco(), c_df* f_df(x, alpha)*b_A(sp, x, alpha)/eps**(1./(3.-alpha)), label='$c_{df}f_{df}b_A$' , color=l.get_c(), linestyle='--')
    plt.xlabel('$r/r_{ISCO}$')

def J(x, alpha):
    """
    Calculates J as in eq (22)
    """
    return 4. * x**(11./2. - alpha) / (1. + x**(3.-alpha))**(7./2.)

def K(x, alpha):
    """
    Calculates K as given by eq (29), but should coincide with (26f) from https://arxiv.org/pdf/1408.3534.pdf
    """
    return (1.+x**(3.-alpha))**(5./2.) * (1. + alpha/3.*x**(3.-alpha)) / (1. + (4.-alpha)*x**(3-alpha) )

def plotPhiprimeprime(sp, r0, r1):
    """
    Plots eq (35) and compares it with the counterpart from the numerical simulation
    """
    r = np.geomspace(r0, r1, num=100)
    alpha = sp.halo.alpha
    eps = F(sp)/Meff(sp)
    x = eps**(1./(3.-alpha))*r
    c_gw, c_df, ctild = coeffs(sp)

    Phipp_ana = Meff(sp)**(1./2.) * eps**(3./2./(3.-alpha)) * c_gw*(1.+ctild*J(x, alpha)*(1.+b_A(sp, x, alpha))) *3./4.* K(x,alpha) * x**(-11./2.)
    plt.loglog(r/sp.r_isco(), Phipp_ana, label=r'$\ddot{\Phi}^{paper}$' )
    #plt.loglog(r/sp.r_isco(), Meff(sp)**(1./2.) * eps**(3./2./(3.-alpha)) \
    #                * (c_gw*f_gw(x, alpha) + c_df*f_df(x, alpha)) * (3. + alpha*x**(3.-alpha))/(x**(5./2.) * (1.+ x**(3.-alpha))**(1./2.) ), label=r'$\ddot{\Phi}^{paper,ref}$' )
    Phipp = (sp.mass(r)/r**3 )**(-1./2.) * (-3.*sp.mass(r)/r**4 + 4.*np.pi *sp.halo.density(r)/r )* inspiral.Classic.da_dt(sp, r)
    plt.loglog(r/sp.r_isco(), Phipp, label=r'$\ddot{\Phi}^{code}$', linestyle='--')

    plt.loglog(r/sp.r_isco(), np.abs(Phipp - Phipp_ana), label=r'$\Delta \ddot{\Phi}$')
    plt.xlabel(r'$r/r_{ISCO}$')


def L(sp, f, accretion=True):
    """
    Calculates L as given by eq (48)
        If accretion=False, then L' as given by eq (58)
    """
    alpha = sp.halo.alpha
    #eps = F(sp)/Meff(sp)
    #c_gw, c_df, ctild = coeffs(sp)
    #c_eps = Meff(sp, 2.*sp.r_isco())**(11./6.-1./3.*alpha) * ctild * eps**((11.-2.*alpha)/(6.-2.*alpha))
    c_eps = 5.*np.pi/32. * Meff(sp)**(-(alpha+5.)/3.) * sp.halo.rho_spike * sp.halo.r_spike**(alpha) * inspiral.Classic.ln_Lambda
    if accretion:
        # TODO: Check prefactor, it's in (36) but not (51)
        b_eps = 4.*(np.pi*f * Meff(sp))**(2./3.) / inspiral.Classic.ln_Lambda * (1. + (np.pi*f * Meff(sp))**(2./3.))
    else:
        b_eps = 0.
    deltatild = (1./np.pi**2 / f**2)**(1.-alpha/3.)
    return 1. + 4.*c_eps*deltatild**((11.-2.*alpha)/(6.-2.*alpha)) * (1. + b_eps)

def mu(sp, f, f_ini):
    """
    Calculates mu as in eq (47) with lower bound f=f_ini
    """
    alpha = sp.halo.alpha
    prefactor = 16.*np.pi * sp.halo.rho_spike* sp.halo.r_spike**alpha * 5./3./np.pi * (8.*np.pi * Meff(sp)**(2./5))**(-5./3.)
    def integrand(y, f):
        return (1. + Meff(sp)**(2./3) * (np.pi**2 *f**2)**(1./3.))  / (Meff(sp)/(np.pi**2 * f**2))**((alpha+1)/3.) / np.pi / f *  f**(-11./3.) / L(sp, f)

    sol = prefactor * odeint(integrand, 0., np.append(f_ini, f), rtol=1e-13, atol=1e-13).flatten()[1:]
    return sp.m2 * np.exp(sol)

'''
def phaseIntegrand(sp, f):
    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp, 2.*sp.r_isco())
    delta = (Meff(sp, 2.*sp.r_isco())/ np.pi**2 / f**2)**((3.-alpha)/3.)
    chi = 1. + delta*eps/3. + (2.-alpha)/9. *delta**2 * eps**2
    x = (delta*eps)**(1./(3.-alpha)) *chi
    c_gw, c_df, ctild = coeffs(sp)
    return chi**(11./2.) / f**(8./3.) / K(x, alpha) / (1. + ctild*J(x,alpha)*(1. + b_A(sp, x, alpha)))
'''

def getPhaseParameters(sp, ev):
    """
    Calculates the terms involved in the derivation of Delta Phi, as the Phi (second part of eq (28b)), 2pi tf (first part of eq(28b), tilde{Phi} (eq(28b)), Delta tilde{Phi} as in eq (56)
    for the solution of the numerical evolution
    """
    omega_s = sp.omega_s(ev.R)
    f_gw = omega_s/np.pi
    f_isco = f_gw[-1]
    #t_c = ev.t[-1] + 5./256. * ev.R[-1]**4/sp.m_total()**2 / sp.m_reduced()
    #if isinstance(t_c, (np.ndarray, collections.Sequence)):
    #    t_c = t_c[-1]

    PhiTild0 =  - 3./4. * (8.*np.pi*sp.m_chirp()*f_gw)**(-5./3.) + 3./4.*(8.*np.pi*sp.m_chirp()*f_isco)**(-5./3.)
    #plt.plot(f_gw*ms.year_to_pc*3.17e-8, PhiTild0, label=r'$\tilde{\Phi}_0^{analytic}$')

    omega_gw = interp1d(ev.t, 2*omega_s, kind='cubic', bounds_error=False, fill_value=(0.,0.) )
    #Phit = np.cumsum([quad(lambda t: omega_gw(t), ev.t[i-1], ev.t[i], limit=500, epsrel=1e-13, epsabs=1e-13)[0] if not i == 0 else 0. for i in range(len(ev.t)) ])
    Phit = odeint(lambda y,t: omega_gw(t), 0., ev.t, atol=1e-13, rtol=1e-13).flatten()
    Phi = Phit - Phit[-1]

    # We use t[-1] as a reference point, because the analytic formalism uses f_isco=f_gw[-1]
    #tpt = 2.*np.pi*f_gw * (ev.t - t_c)
    tpt = 2.*np.pi*f_gw * (ev.t - ev.t[-1])

    PhiTild = tpt - Phi
    DeltaPhi = PhiTild - PhiTild0
    return f_gw, Phi, tpt, PhiTild, DeltaPhi


def plotPhase(sp, ev_acc, ev_nacc):
    """
    Plots the different terms of the derivation of Delta tilde{Phi} semianalytically and compares them to the numerical evolution. Additionally, calculates tilde{Phi}_1 as in eq (57) for both paper and code and compares the delta tilde{Phi} as given by eq (59)
    """
    # Code accretion
    sp.m2 = ev_acc.m2
    f_gw, Phi, tpt, PhiTild, DeltaPhi = getPhaseParameters(sp, ev_acc)
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, Phi, label=r'$\Phi^{code}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, tpt, label=r'$2\pi t^{code}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, PhiTild, label=r'$\tilde{\Phi}^{code}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, DeltaPhi, label=r'$\Delta\tilde{\Phi}^{code}$')
    sp.m2 = ev_nacc.m2

    # Paper accretion
    f_isco = f_gw[-1]
    PhiTild0 = - 3./4.*(8.*np.pi*sp.m_chirp()*f_gw)**(-5./3.) + 3./4.*(8.*np.pi*sp.m_chirp()*f_isco)**(-5./3.)

    mu_interp = interp1d(f_gw, mu(sp, f_gw, f_gw[0]), kind='cubic', bounds_error=False, fill_value='extrapolate')
    #Phi_ana = np.cumsum([quad(lambda f: f**(-8./3.)/L(sp, f)/mu_interp(f), f_gw[i-1], f_gw[i], limit=200, epsrel=1e-13, epsabs=1e-13)[0] if not i == 0 else 0. for i in range(len(f_gw)) ])
    Phi_ana = solve_ivp(lambda f,y: f**(-8./3.)/L(sp, f)/mu_interp(f), [f_gw[0], f_gw[-1]], [0.], t_eval=f_gw, atol=1e-13, rtol=1e-13).y[0]
    Phi_ana = 10./3. * (8.*np.pi*Meff(sp)**(2./5.))**(-5./3.) * (Phi_ana - Phi_ana[-1])
    #tpt_ana = np.cumsum([quad(lambda f: f**(-11./3.)/L(sp, f)/mu_interp(f), f_gw[i-1], f_gw[i], limit=200, epsrel=1e-13, epsabs=1e-13)[0] if not i==0 else 0. for i in range(len(f_gw)) ])
    tpt_ana = solve_ivp(lambda f,y: f**(-11./3.)/L(sp, f)/mu_interp(f), [f_gw[0], f_gw[-1]], [0.], t_eval=f_gw, atol=1e-13, rtol=1e-13).y[0]
    tpt_ana = 10./3. * (8.*np.pi*Meff(sp)**(2./5.))**(-5./3.) * f_gw * ( tpt_ana - tpt_ana[-1])
    PhiTild_ana = tpt_ana - Phi_ana
    DeltaPhi_ana = PhiTild_ana - PhiTild0

    plt.plot(f_gw*ms.year_to_pc*3.17e-8, Phi_ana, label=r'$\Phi^{paper}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, tpt_ana, label=r'$2\pi t^{paper}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, PhiTild_ana, label=r'$\tilde{\Phi}^{paper}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, DeltaPhi_ana, label=r'$\Delta\tilde{\Phi}^{paper}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, np.abs(DeltaPhi - DeltaPhi_ana), label=r'$\Delta \Delta\tilde{\Phi}$')

    # Code no accretion
    f_gw_nacc, _, __, ___, DeltaPhi_nacc = getPhaseParameters(sp, ev_nacc)
    DeltaPhi_naccinterp = interp1d(f_gw_nacc, DeltaPhi_nacc, kind='cubic', bounds_error=False, fill_value=(0.,0.))
    deltaPhi = np.abs(DeltaPhi_naccinterp(f_gw) - DeltaPhi)
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, deltaPhi, label=r'$\delta\tilde{\Phi}^{code}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, deltaPhi/DeltaPhi, label=r'$\delta\tilde{\Phi}^{code}/\Delta\tilde{\Phi}$')

    # Paper no accretion
    Phi_ana = np.cumsum([quad(lambda f: f**(-8./3.)/L(sp, f, accretion=False), f_gw_nacc[i-1], f_gw_nacc[i], limit=200, epsrel=1e-13, epsabs=1e-13)[0] if not i == 0 else 0. for i in range(len(f_gw_nacc)) ])
    #Phi_ana = solve_ivp(lambda f,y: f**(-8./3.)/L(sp, f, accretion=False), [f_gw_nacc[0], f_gw_nacc[-1]], [0.], t_eval=f_gw_nacc, atol=1e-13, rtol=1e-13, method='LSODA').y[0]
    Phi_ana = 10./3. * (8.*np.pi*sp.m_chirp())**(-5./3.) * (Phi_ana - Phi_ana[-1])
    tpt_ana = np.cumsum([quad(lambda f: f**(-11./3.)/L(sp, f, accretion=False), f_gw_nacc[i-1], f_gw_nacc[i], limit=200, epsrel=1e-13, epsabs=1e-13)[0] if not i==0 else 0. for i in range(len(f_gw_nacc)) ])
    #tpt_ana = solve_ivp(lambda f,y: f**(-11./3.)/L(sp, f, accretion=False), [f_gw_nacc[0], f_gw_nacc[-1]], [0.], t_eval=f_gw_nacc, atol=1e-13, rtol=1e-13, method='LSODA').y[0]
    tpt_ana = 10./3. * (8.*np.pi*sp.m_chirp())**(-5./3.) * f_gw_nacc * ( tpt_ana - tpt_ana[-1])
    PhiTild_nacc_ana = tpt_ana - Phi_ana
    PhiTild_nacc_anaInterp = interp1d(f_gw_nacc, PhiTild_nacc_ana, kind='cubic', bounds_error=False, fill_value=(0.,0.))
    deltaPhi_ana = np.abs(PhiTild_nacc_anaInterp(f_gw) - PhiTild_ana)

    plt.plot(f_gw*ms.year_to_pc*3.17e-8, deltaPhi_ana, label=r'$\delta\tilde{\Phi}^{paper}$')
    plt.plot(f_gw*ms.year_to_pc*3.17e-8, deltaPhi_ana/DeltaPhi_ana, label=r'$\delta\tilde{\Phi}^{paper}/\Delta\tilde{\Phi}$')

    plt.xlabel('f')
    plt.xscale('log')
    plt.yscale('symlog')
    #plt.yscale('log')


def plotWaveform(sp, ev):
    """
    Plots the gravitational waveform of h as given by eq (40) and compares them to the code
    """
    f_gw, h, _, Psi  = waveform.h_2( sp, ev)
    plt.loglog(f_gw*ms.year_to_pc*3.17e-8, h, label=r'$\tilde{h}^{code}$')

    alpha = sp.halo.alpha
    eps = F(sp,2.*sp.r_isco())/Meff(sp)
    A = (5./24.)**(1./2.) * np.pi**(-2./3.) /sp.D * sp.m_chirp()**(5./6.)
    plt.loglog(f_gw*ms.year_to_pc*3.17e-8, A*f_gw**(-7./6.) * (L(sp,f_gw))**(-1./2.), label=r'$\tilde{h}^{paper,approx}$')

    delta = (Meff(sp)/np.pi**2 / f_gw**2)**(1.-alpha/3.)
    chi = 1. + delta*eps/3. + (2.-alpha)/9. *delta**2 * eps**2
    x = (delta*eps)**(1./(3.-alpha)) *chi
    c_gw, c_df, ctild = coeffs(sp)
    plt.loglog(f_gw*ms.year_to_pc*3.17e-8, A*f_gw**(-7./6.) * chi**(19./4.) * (K(x, alpha)* (1. + ctild*J(x, alpha)*(1.+b_A(sp, x, alpha)) ))**(-1./2.), label=r'$\tilde{h}^{paper}$' )
    plt.ylabel('h'); plt.xlabel('f')


m1 = 1e5 *ms.solar_mass_to_pc
m2 = 1. *ms.solar_mass_to_pc
D = 1e3
sp_1 = ms.SystemProp(m1, m2, halo.Spike( 226*ms.solar_mass_to_pc, 0.54, 7./3.), D)


plt.figure()
plotDiffEq(sp_1, sp_1.r_isco(), 1e7*sp_1.r_isco())
plt.legend(); plt.grid()

plt.figure()
plotPhiprimeprime(sp_1, sp_1.r_isco(), 1e5*sp_1.r_isco())
plt.legend(); plt.grid()

R0 = 100.*sp_1.r_isco()
ev_nacc = inspiral.Classic.evolve_circular_binary(sp_1, R0, sp_1.r_isco(), acc=1e-13, accretion=False)
ev_acc = inspiral.Classic.evolve_circular_binary(sp_1, R0, sp_1.r_isco(), acc=1e-13, accretion=True)

plt.figure()
plotPhase(sp_1, ev_acc, ev_nacc)
plt.legend(); plt.grid()

plt.figure()
plotWaveform(sp_1, ev_acc)
plt.legend(); plt.grid()

plt.figure()
mu_ana = mu(sp_1, sp_1.omega_s(ev_acc.R)/np.pi, sp_1.omega_s(ev_acc.R[0])/np.pi)
plt.loglog(ev_acc.t,  mu_ana/m2 -1., label='$\Delta m_2^{paper}/m_2$')
plt.loglog(ev_acc.t, ev_acc.m2/m2 - 1., label="$\Delta m_2^{code}/m_2$", linestyle='--')
plt.loglog(ev_acc.t, np.abs(mu_ana - ev_acc.m2)/m2, label="$\Delta m_2$", linestyle='--')

plt.legend(); plt.grid()
print("mass increase:", ev_acc.m2[-1]/ev_acc.m2[0] -1.)

'''
# Now check the eccentric implementation with a tiny eccentricity, it should be very similar
a0 = 100.*sp_1.r_isco()
e0 = 1e-5
sp_1.m2 = 1.*ms.solar_mass_to_pc

#t2, a2, e2, m22 = inspiral.Classic.evolve_elliptic_binary(sp_1, a0, e0, sp_1.r_isco(), acc=1e-12, accretion=True)
ev2 = inspiral.Classic.evolve_elliptic_binary(sp_1, a0, e0, sp_1.r_isco(), acc=1e-12, accretion=True)

plt.figure()
plt.loglog(ev_acc.t, ev_acc.R, label='R, circular')
plt.loglog(ev2.t, ev2.a, label='a, elliptic')

plt.loglog(ev_acc.t, ev_acc.m2, label='$m_2$, circular')
plt.loglog(ev2.t, ev2.m2, label='$m_2$, elliptic')

plt.loglog(ev2.t, ev2.e, label='e')
plt.grid(); plt.legend()
'''

plt.show()



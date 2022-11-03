import numpy as np
import matplotlib.pyplot as plt
import imripy.halo
from scipy.integrate import odeint, quad
from scipy.interpolate import interp1d
import time

'''
def CompareMassFunction(halo, r_grid, ax):
    tic = time.perf_counter()
    m_ana = halo.mass(r_grid)  # Here we call the analytic expression
    toc = time.perf_counter()
    t_ana = toc - tic

    tic = time.perf_counter()
    m_num = imripy.halo.MatterHalo.mass(halo, r_grid) # Here we call the numerical integration in the base class
    toc = time.perf_counter()
    t_num = toc - tic

    print("Comparing implementation for " + str(halo))
    print("elapsed time in analytic case: ", t_ana)
    print("elapsed time in numeric case:", t_num)
    print("The average relative error is ", np.average(np.abs(m_num/m_ana - 1.)))
    print("The maximal relative error is ", np.max(np.abs(m_num/m_ana - 1.)))

    l, = ax.loglog(r_grid, m_ana, alpha=0.5, label=str(halo) + ',analytic')
    ax.loglog(r_grid, m_num, color=l.get_c(), linestyle='--', label=str(halo) + ',numeric')


def TestEddingtonInversion(halo, r_grid, ax_r, ax_eps, extPotential=None, f_ana=None, pot_ana=None):
    if extPotential is None:
        integrand = lambda r, m: halo.mass(np.abs(r))/r**2
        Phi_inf = quad(integrand, r_grid[-1], np.inf, args=(0.), limit=200)[0]
        Phi_inf = np.clip(Phi_inf, 1e-50, None)
        extPotential =  (odeint(integrand, Phi_inf, -r_grid[::-1], tfirst=True, atol=1e-10, rtol=1e-10)[::-1,0])
        extPotential = interp1d(r_grid, extPotential, kind='cubic', bounds_error=False, fill_value=(0.,0.))

    if not pot_ana is None:
        l, = ax_r.loglog(r_grid, extPotential(r_grid), linestyle='--', label=str(halo) + ' $\Phi$, recovered')
        ax_r.loglog(r_grid, pot_ana(r_grid), color=l.get_c(), alpha=0.5, label=str(halo) + ' $\Phi$, analytic')

    Eps_grid = np.geomspace(extPotential(r_grid[-1]), extPotential(r_grid[0]), 500)
    haloRec = imripy.halo.DynamicSS.FromStatic(Eps_grid, halo, extPotential)
    l, = ax_r.loglog(r_grid, halo.density(r_grid), alpha=0.5, label=str(halo) + ',static')
    ax_r.loglog(r_grid, haloRec.density(r_grid), color=l.get_c(), linestyle='--', label=str(halo) + ',recovered')
    ax_eps.loglog(Eps_grid, haloRec.f_grid, color=l.get_c(), linestyle='--', label=str(halo) +',recovered')
    if not f_ana is None:
        ax_eps.loglog(Eps_grid, f_ana(Eps_grid), color=l.get_c(), alpha = 0.5 , label=str(halo) + ',analytic')
    return haloRec


n = 1000
r_grid = np.geomspace(1e-5, 1e5, n)

# Test numerical and analytical mass functions
ax = plt.gca()
CompareMassFunction(imripy.halo.ConstHalo(1.), r_grid, ax)
CompareMassFunction(imripy.halo.NFW(1., 1e3), r_grid, ax)
CompareMassFunction(imripy.halo.SpikedNFW(1., 1e3, 1e-2, 7./3.), r_grid, ax)
CompareMassFunction(imripy.halo.Spike(1., 1e-2, 7./3.), r_grid, ax)
CompareMassFunction(imripy.halo.Hernquist(1., 1e-3), r_grid, ax)
ax.set_xlabel("r")
ax.set_ylabel("m")
plt.grid(); plt.legend()


fig, (ax_r, ax_eps) = plt.subplots(2, 1, figsize=(20,20))
#extPotential = lambda r: 1/r

Spike = imripy.halo.Spike(1., 1e-2, 7./3.)
extPotential = lambda r : 1./r
from scipy.special import gamma
f_ana = lambda E : Spike.rho_0 * Spike.alpha*(Spike.alpha-1.)/(2.*np.pi)**(3./2.) * (Spike.r_spike/1.)**Spike.alpha * gamma(Spike.alpha-1.)/gamma(Spike.alpha-1./2.) * E**(Spike.alpha-3./2.)
TestEddingtonInversion( Spike , r_grid, ax_r, ax_eps, extPotential, f_ana)

Hern = imripy.halo.Hernquist(1., 1e2)
def f_ana_Hern(Eps):
    M = 2.*np.pi * Hern.rho_0 * Hern.r_s**3
    E = Eps*Hern.r_s/M
    return (M * Hern.r_s)**(-3./2.)/np.sqrt(2)/(2.*np.pi)**3 * np.sqrt(E)/(1-E)**2  *( (1.-2*E)*(8.*E**2 - 8.*E - 3.) + 3.*np.arcsin(np.sqrt(E)) / np.sqrt(E*(1-E))  )

def pot_ana_Hern(r):
    M = 2.*np.pi * Hern.rho_0 * Hern.r_s**3
    return M/(r + Hern.r_s)

Hern_rec = TestEddingtonInversion( Hern , r_grid, ax_r, ax_eps, extPotential=None, f_ana=f_ana_Hern, pot_ana=pot_ana_Hern)

Eps_grid = Hern_rec.Eps_grid
Hern_ana = imripy.halo.DynamicSS(Eps_grid, f_ana_Hern(Eps_grid), pot_ana_Hern)
ax_eps.loglog(Eps_grid, f_ana_Hern(Eps_grid), label='Hern2')
ax_r.loglog(r_grid, Hern_ana.density(r_grid), label='Hern2')
ax_eps.grid(); ax_eps.legend(); ax_eps.set_xlabel(r"$\varepsilon$")
ax_r.grid(); ax_r.legend(); ax_r.set_xlabel("r")

'''

plt.figure(figsize=(12,6))
r_grid = np.geomspace(1., 100, 30)
m = 1.
m_prime = 10.


spike = imripy.halo.Spike(1., 0.1, 7./3.)
adf_spike = imripy.halo.AnalyticDistributionFunction.FromSpike(m, spike)
adf_spike.relativisticCalculation=True

newPotential = lambda r : m_prime/r
Eps_grid = np.geomspace(np.sqrt(8./9), 1., 100)
L_grid = np.geomspace(1e-5, 1e5, 100)
Eps_grid, L_grid = np.meshgrid(Eps_grid, L_grid)
print(Eps_grid, L_grid, len(L_grid))

#Eps_grid_prime, I_r, I_r_prime = imripy.halo.DynamicDistributionFunction.adiabaticGrowthRelativistic(Eps_grid, L_grid, adf_spike.potential, newPotential)

#print(Eps_grid, Eps_grid_prime)
'''
ax = plt.figure(figsize=(20,15)).add_subplot(projection='3d')
ax.plot_wireframe(Eps_grid, L_grid, I_r)
ax.plot_wireframe(Eps_grid, L_grid, I_r_prime, color='red')

plt.figure()
'''
ddf_spike = imripy.halo.DynamicDistributionFunction.FromAnalyticDistributionFunction(adf_spike, Eps_grid, L_grid=L_grid)
ddf_grown = imripy.halo.DynamicDistributionFunction.FromAdiabaticGrowth(ddf_spike, newPotential, newtonian=False, Eps_grid=Eps_grid, L_grid=L_grid)
plt.show()

t0 = time.perf_counter()
a = adf_spike.f(Eps_grid[3]*1.1)
t1 = time.perf_counter()
b = ddf_spike.f(Eps_grid[3]*1.1)
t2 = time.perf_counter()
c = ddf_grown.f(Eps_grid[3]*1.1)
t3 = time.perf_counter()
print(f"a = {a}, b = {b}, c={c}  took {t1-t0}s vs {t2-t1}s vs {t3-t2}s")

plt.loglog(r_grid, adf_spike.density(r_grid), label="adf")
print("1")
plt.loglog(r_grid, ddf_spike.density(r_grid), label="ddf")
print("2")
plt.loglog(r_grid, ddf_grown.density(r_grid), label="ddf+grown")

plt.grid(); plt.legend()
plt.show()


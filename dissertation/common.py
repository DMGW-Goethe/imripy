import numpy as np

from scipy.integrate import simpson

from imripy import constants as c, kepler, merger_system as ms, halo, inspiral
from imripy.inspiral import forces

"""
This file defines the common environmental effects that we use and their parametrization.
"""

def AccretionDisk(hs):
    m1 = hs.m1
    alpha = 0.1
    f_edd = 0.1
    Mdot_edd = (2.2 * 1e-9 * m1 /0.3064)

    dmDisk = halo.DerdzinskiMayerDisk(m1, f_edd*Mdot_edd, alpha)
    r_grid = np.geomspace(1.5*hs.r_isco, 1e7*hs.r_isco, 1000)
    dmDisk_int = dmDisk.CreateInterpolatedHalo(r_grid)
    gdf_dm = inspiral.forces.GasDynamicalFriction(disk=dmDisk_int)
    return dmDisk_int, gdf_dm

def DMSpike(hs, gamma):
    m1 = hs.m1
    spike = halo.Spike.FromSpikedNFW(halo.SpikedNFW.FromNFW(halo.NFW.FromHaloMass(1e3*hs.m1, 20), m1, gamma))

    df = inspiral.forces.DynamicalFriction(halo=spike, haloPhaseSpaceDescription=True, includeHigherVelocities=True,
                                             relativisticCorrections=True)
    return spike, df


def StellarDistribution(hs, E_m_s = c.solar_mass_to_pc):
    m1 = hs.m1
    rho_st = 1.
    r_st = 11. * (m1/1e8 / c.solar_mass_to_pc)**(0.58)
    alpha = 7./4.
    stellarHalo = halo.Spike(rho_st, r_st, alpha, m1)
    stellarHalo.rho_spike *= m1 / stellarHalo.mass(r_st)
    stellarDiffusion = inspiral.forces.StellarDiffusion(hs, stellarHalo, E_m_s=E_m_s)
    return stellarHalo, stellarDiffusion

def StellarDistributionAna(hs, E_m_s = c.solar_mass_to_pc):
    m1 = hs.m1
    rho_st = 1.
    r_st = 11. * (m1/1e8 / c.solar_mass_to_pc)**(0.58)
    alpha = 7./4.
    stellarHalo = halo.Spike(rho_st, r_st, alpha, m1)
    stellarHalo.rho_spike *= m1 / stellarHalo.mass(r_st)
    stellarDiffusion = inspiral.forces.StellarDiffusionAna(hs, stellarHalo, E_m_s=E_m_s)
    return stellarHalo, stellarDiffusion


def energy_loss(hs, ev, df, opt):
    dE = np.array([df.dE_dt(hs, ev.get_kepler_orbit(i), opt) for i in range(len(ev.t))])
    return simpson(dE, x= ev.t)

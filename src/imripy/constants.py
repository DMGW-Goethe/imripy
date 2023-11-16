'''
This file is supposed to provide some constant to convert to the geometrized units that the code uses.
We use c = G = 1 and convert all relevant scales to parsec pc.
Therefore the unit of mass is [M] = pc, the unit of time and length [t] = [l] = pc,
the unit of frequency [f] = 1/pc, the unit of density [rho] = 1/pc^2
and so on...
These constants should provide an easy multiplicative conversion factor,
such that e.g. M_sun = 1. * solar_mass_to_pc would be the solar mass in pc
'''
# Herz to 1/pc
hz_to_invpc = 1.029e8
# seconds to pc
s_to_pc = 9.716e-9
# years to pc
year_to_pc = 0.3064
# meter to pc
m_to_pc = 3.241e-17
# solar mass to pc
solar_mass_to_pc = 4.8e-14
# gram / cubic centimeter (density) to 1/pc^2
g_cm3_to_invpc2 = 7.072e8
# gram / centimeter squared (surface density) to 1/pc
g_cm2_to_invpc = 7.426e-27 / m_to_pc
# Giga electron volt / cubic centimeter (density) to 1/pc^2
GeV_cm3_to_invpc2 = 1.783e-24 * g_cm3_to_invpc2
# rad to arcmin
rad_to_arcmin = 3437.8

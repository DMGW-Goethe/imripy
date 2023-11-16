import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from scipy.interpolate import interp1d

from imripy import halo, inspiral, waveform, constants as c, merger_system as ms, kepler

def plotEvolution(hs, ev, ax_a=None, ax_e=None, label="", ax_ae=None, ax_1mea=None, ax_m=None, m2=1., ax_pa=None, ax_ia=None, ax_n=None, color=None, linestyle=None):
    """
    Plots the evolution of the system in the natural units that are used throughout the code.
    The evolution can be plotted as semimajor axis / time, eccentricity / time, eccentricity / semimajor axis, or relative mass/time,
        depending on which plt.axes objects are passed.
    The lines will have the same color and label

    Parameters:
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
        ev (inspiral.Classic.Evolution) : The evolution object that results from the inspiral modeling
        ax_a (plt.axes)     (optional)  : The axes on which to plot semimajor axis / time
        ax_e (plt.axes)     (optional)  : The axes on which to plot eccentricity / time
        ax_ae (plt.axes)    (optional)  : The axes on which to plot eccentricity / semimajor axis
        ax_1mea (plt.axes)  (optional)  : The axes on which to plot semimajor axis / (1-eccentricity)
        ax_m (plt.axes)     (optional)  : The axes on which to plot relative mass / time
        m2   (float)        (optional)  : The initial mass of the system
        ax_pa (plt.axes)    (optional)  : The axis on which to plot periapse angle / time
        ax_ia (plt.axes)    (optional)  : The axis on which to plot inclination angle / time
        ax_n (plt.axes)     (optional)  : The axes on which to plot the braking index / frequency
        label (string)      (optional)  : The label corresponding to the lines

    Returns:
        out : matplotlib.lines.Line2D
            The last line object plotted
    """
    if not ax_a is None:
        l, = ax_a.loglog(ev.t/c.year_to_pc, ev.a/hs.r_isco, label=label, color=color, linestyle=linestyle)
        color = l.get_c()
    if not ax_e is None:
        l, = ax_e.loglog(ev.t/c.year_to_pc, ev.e, color=color, label=label, linestyle=linestyle)
        color = l.get_c()
    if not ax_m is None:
        l, = ax_m.loglog(ev.t/c.year_to_pc, ev.m2/m2-1., color=color, label=label, linestyle=(linestyle if not linestyle is None else ':'))
        color = l.get_c()
    if not ax_ae is None:
        l, = ax_ae.plot(ev.a/hs.r_isco, ev.e, color=color, label=label, linestyle=linestyle)
        color = l.get_c()
    if not ax_pa is None:
        l, = ax_pa.plot(ev.t[np.where(ev.e > 0)]/c.year_to_pc, ev.periapse_angle[np.where(ev.e)]*c.rad_to_arcmin, color=color, label=label, linestyle=linestyle)
        if np.any(ev.e == 0.):
            first = np.where(ev.e == 0.)[0][0]
            ax_pa.plot(ev.t[first]/c.year_to_pc, ev.periapse_angle[first]*c.rad_to_arcmin, marker='o', color=color)
        color = l.get_c()
    if not ax_ia is None:
        l, = ax_ia.plot(ev.t/c.year_to_pc, ev.inclination_angle*c.rad_to_arcmin, color=color, label=label, linestyle=linestyle)
        color = l.get_c()
    if not ax_1mea is None:
        l, = ax_1mea.loglog(1. - ev.e, ev.a/hs.r_isco, color=color, label=label, linestyle=linestyle)
        color = l.get_c()
    if not ax_n is None:
        F, n = waveform.BrakingIndex(hs, ev)
        l, = ax_n.plot(2*F[2:-2]/c.hz_to_invpc, n[2:-2], color=color, label=label, linestyle=linestyle)
        color = l.get_c()
    return l

def plotGWcharacteristicStrain(hs, ev, ax_h, label="", acc=1e-13, harmonics=[2], color=None, **kwargs):
    """
    Plots the characteristic strain in its harmonics in the natural units that are used throughout the code.
    The lines will have the same color and alternating linestyle

    Parameters:
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
        ev (inspiral.Classic.Evolution) : The evolution object that results from the inspiral modeling
        ax_h (plt.axes)                 : The axes on which to plot characteristic strain / frequency
        harmonics (list of integers)    : The list of harmonics to be plotted
        label (string)      (optional)  : The label corresponding to the lines
        color               (optional)  : The color of the lines. Can be anything that matplotlib accepts as color
        **kwargs                        : Other parameters that can be passed to the plotting
    """
    linecycler = cycle(["-", "--", "-.", ":"])
    color_cycle = ax_h._get_lines.prop_cycler
    l = None
    col = color if not color is None else next(color_cycle)['color']
    for n in harmonics:
        wf = waveform.h_n(n, hs, ev, acc=acc)
        l, = ax_h.loglog(wf[0]/c.hz_to_invpc, 2.*wf[0]*np.abs(wf[1]), linestyle=next(linecycler),
                                 label=label, color=col, **kwargs)


def plotDeltaN(hs, ev_0, ev_1, ax_dN, ax_di=None, n=2, acc=1e-13, plotFgw5year=False, min_dN=10., **kwargs):
    """
    Plots the dephasing of a given harmonic in the natural units that are used throughout the code.

    Parameters:
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
        ev_0 (inspiral.Classic.Evolution) : The evolution object that results from the baseline inspiral modeling
        ev_1 (inspiral.Classic.Evolution) : The evolution object that results from the alternative inspiral modeling
        ax_dN (plt.axes)                  : The axes on which to plot the difference in cycles left to observe
        ax_di (plt.axes)    (optional)    : The axes on which to plot the dephasing index
        n (int)             (optional)    : The harmonic to be plotted
        plotFgw5year (bool) (optional)    : Whether to plot the line that represents 5 years to merger in vacuum case
        min_dN (float)      (optional)    : The minimum dephasing amount for which to plot the dephasing index
        **kwargs                          : Other parameters that can be passed to the plotting, i.e. label, color, linestyle

    Returns:
        f_gw1 : array_like
            The frequencies corresponding to the difference in cycles
        dN   : array_like
            The difference in cycles of the two systems
    """
    f_gw0, N_0 = waveform.N_cycles_n(n, hs, ev_0, acc=acc)
    N_0interp = interp1d(f_gw0, N_0, kind='cubic', bounds_error=False, fill_value=(0.,0.))
    f_gw1, N_1 = waveform.N_cycles_n(n, hs, ev_1, acc=acc)

    dN = N_1 - N_0interp(f_gw1)
    l, = ax_dN.loglog(f_gw1/c.hz_to_invpc, np.abs(dN), **kwargs)

    if plotFgw5year:
        f_gw5yr = interp1d(ev_1.t, f_gw1, kind='cubic', bounds_error=True)(ev_1.t[-1] - 5.*c.year_to_pc)
        ax_dN.axvline(f_gw5yr/c.hz_to_invpc, linestyle='--', color=l.get_c())

    if not ax_di is None:
        ddN_df = np.gradient(dN, f_gw1)
        stop = np.where(np.abs(dN) < min_dN)[0]
        stop = stop[0] if len(stop) > 0 else len(ddN_df)
        if 'color' in kwargs:
            del kwargs['color']
        # print(f_gw1, N_1, dN, ddN_df, ddN_df/dN * f_gw1)
        ax_di.plot(f_gw1[:stop]/c.hz_to_invpc, (ddN_df/dN * f_gw1)[:stop], color=l.get_c(), **kwargs)

    return f_gw1, dN


def streamline(ax, hs, opt, m2, a_grid, j_grid, cmap='plasma'):
    """
    Plots the streamlines and their strength of the ode system in a and j space.
    a is the semimajor axis and j the specific angular momentum.
    The dissipative forces considered are to be given to opt

    Parameters:
        ax (plt.axes)                   : The axes on which to plot
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
        opt (inspiral.Classic.EvolutionOptions) : The evolution options. The important part is opt.dissipativeForces
        m2  (float)                     : Mass of the secondary
        a_grid (np.array)               : The grid in semimajor axis a
        j_grid (np.array)               : The grid in specific angular momemtum j
        cmap (plt colormap) (optional)  : A plt colormap or the identifying name

    Returns:
        out : matplotlib.image.AxesImage
            The axes image plotted
    """
    na = len(a_grid); ne = len(j_grid)

    a, j = np.meshgrid(a_grid, j_grid)
    e = np.sqrt(1. - j**2)
    e_grid = np.sqrt(1. - j_grid**2)
    da_grid = np.zeros(np.shape(a))
    dj_grid = np.zeros(np.shape(j))

    ko = kepler.KeplerOrbit(hs, m2, 0., prograde=opt.progradeRotation)
    # Calculate Derivatives
    for i in range(na):
        for k in range(ne):
            if a_grid[i]* (1.-e[k,i]) < 6.*hs.m1: # A bit smaller than the BH to have no alialising effects at the Loss Cone
                continue
            ko.a = a_grid[i]; ko.e = e[k,i]
            da_grid[k,i], dE_dt = inspiral.Classic.da_dt(hs, ko, opt=opt, return_dE_dt=True)
            dj_grid[k,i] = -e[k,i]/j[k,i] * inspiral.Classic.de_dt(hs, ko, dE_dt=dE_dt, opt=opt)

    dloga = da_grid/a
    dlogj = dj_grid/j
    speed = np.sqrt(dlogj**2 + dloga**2)

    # Plot Streamlines & Strength
    im = ax.imshow(speed.T, norm='log',
                   extent = [np.log10(j_grid[0]), np.log10(j_grid[-1]), np.log10(a_grid[0]/hs.r_isco), np.log10(a_grid[-1]/hs.r_isco)],
                  origin = 'lower', interpolation='bilinear', aspect='auto', cmap=cmap, zorder=1)
    strm = ax.streamplot(np.log10(j_grid), np.log10(a_grid/hs.r_isco),
                  dlogj.T, dloga.T, color='black', zorder=1)
    # Cover Loss cone
    ax.plot(np.log10(j_grid), np.log10(8./6./(1.-e_grid)), color='black', zorder=2)
    ax.fill_between(np.log10(j_grid), np.log10(8./6./(1.-e_grid)), color='gray', alpha=1., zorder=2)
    return im

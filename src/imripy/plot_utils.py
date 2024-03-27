import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from scipy.interpolate import interp1d

from imripy import halo, inspiral, waveform, constants as c, merger_system as ms, kepler

def plotEvolution(hs, ev, ax_a=None, ax_e=None, label="", ax_ae=None, ax_1mea=None, ax_m=None, m2=1., ax_pa=None, ax_ia=None, ax_n=None, color=None, linestyle=None, **kwargs):
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
        l, = ax_a.loglog(ev.t/c.year_to_pc, ev.a/hs.r_isco, label=label, color=color, linestyle=linestyle, **kwargs)
        color = l.get_c()
    if not ax_e is None:
        l, = ax_e.loglog(ev.t/c.year_to_pc, ev.e, color=color, label=label, linestyle=linestyle, **kwargs)
        color = l.get_c()
    if not ax_m is None:
        l, = ax_m.loglog(ev.t/c.year_to_pc, ev.m2/m2-1., color=color, label=label, linestyle=(linestyle if not linestyle is None else ':'), **kwargs)
        color = l.get_c()
    if not ax_ae is None:
        l, = ax_ae.plot(ev.a/hs.r_isco, ev.e, color=color, label=label, linestyle=linestyle, **kwargs)
        color = l.get_c()
    if not ax_pa is None:
        l, = ax_pa.plot(ev.t[np.where(ev.e > 0)]/c.year_to_pc, ev.periapse_angle[np.where(ev.e)]*c.rad_to_arcmin, color=color, label=label, linestyle=linestyle, **kwargs)
        if np.any(ev.e == 0.):
            first = np.where(ev.e == 0.)[0][0]
            ax_pa.plot(ev.t[first]/c.year_to_pc, ev.periapse_angle[first]*c.rad_to_arcmin, marker='o', markerfacecolor=color, **kwargs)
        color = l.get_c()
    if not ax_ia is None:
        l, = ax_ia.plot(ev.t/c.year_to_pc, ev.inclination_angle*c.rad_to_arcmin, color=color, label=label, linestyle=linestyle, **kwargs)
        color = l.get_c()
    if not ax_1mea is None:
        l, = ax_1mea.loglog(1. - ev.e, ev.a/hs.r_isco, color=color, label=label, linestyle=linestyle, **kwargs)
        color = l.get_c()
    if not ax_n is None:
        F, n = waveform.BrakingIndex(hs, ev)
        l, = ax_n.plot(2*F[2:-2]/c.hz_to_invpc, n[2:-2], color=color, label=label, linestyle=linestyle, **kwargs)
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
                                 label=label.format(n) if "{0}" in label else label, color=col, **kwargs)


def plotLastTyears(hs, ev, ax, t=5.*c.year_to_pc, n=2, marker=None, y=None, **kwargs):
    """
    Plots a line or a marker at the last t years to inspiral

    Parameters:
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
        ev (inspiral.Classic.Evolution) : The evolution object that results from the inspiral modeling
        ax (plt.axes)                   : The axes object to plot on
        n  (int)                        : Which harmonic to take the frequency of
        t  (float)                      : The time in code units (pc)
        marker (plt.marker)             : The marker style for matplotlib, e.g. 'o', or 'p'
        y (function)                    : The function that gives the y value to plot the marker at
        **kwargs                        : Other parameters that can be passed to the plotting, i.e. label, color, linestyle

    Returns:
        f_t : float
            The corresponding frequency
    """
    f = n * np.sqrt(ev.m_tot/ev.a**3) / 2./np.pi
    t_to_f = interp1d(ev.t, f, kind='cubic', bounds_error=False, fill_value=(0.,0.))
    f_t = t_to_f(ev.t[-1] - t)
    if f_t == 0.:
        return None, f_t
    if marker is None:
        l = ax.axvline(f_t/c.hz_to_invpc, linestyle='--', **kwargs)
    else:
        l, = ax.plot(f_t/c.hz_to_invpc, y(f_t), marker=marker, **kwargs)
    return l, f_t


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
        plotFgw5year (bool) (optional)    : Whether to plot the marker that represents 5 years to merger
        min_dN (float)      (optional)    : The minimum dephasing amount for which to plot the dephasing index
        **kwargs                          : Other parameters that can be passed to the plotting, i.e. label, color, linestyle

    Returns:
        f_gw1 : array_like
            The frequencies corresponding to the difference in cycles
        dN   : array_like
            The difference in cycles of the two systems
    """
    ev_0.f_gw, ev_0.N = waveform.N_cycles_n(n, hs, ev_0, acc=acc)
    N_0interp = interp1d(ev_0.f_gw, ev_0.N, kind='cubic', bounds_error=False, fill_value=(0.,0.))
    ev_1.f_gw, ev_1.N = waveform.N_cycles_n(n, hs, ev_1, acc=acc)

    dN = ev_1.N - N_0interp(ev_1.f_gw)
    l, = ax_dN.loglog(ev_1.f_gw/c.hz_to_invpc, np.abs(dN), **kwargs)

    if plotFgw5year:
        l, f5yrs = plotLastTyears(hs, ev_1, ax_dN, t=5.*c.year_to_pc, n=n, marker='p', y=interp1d(ev_1.f_gw, np.abs(dN)), color=l.get_c())

    if not ax_di is None:
        ddN_df = np.gradient(dN, ev_1.f_gw)
        stop = np.where(np.abs(dN) < min_dN)[0]
        stop = stop[0] if len(stop) > 0 else len(ddN_df)
        if 'color' in kwargs:
            del kwargs['color']
        # print(f_gw1, N_1, dN, ddN_df, ddN_df/dN * f_gw1)
        ax_di.plot(ev_1.f_gw[:stop]/c.hz_to_invpc, (ddN_df/dN * ev_1.f_gw)[:stop], color=l.get_c(), **kwargs)

    return ev_1.f_gw, dN



def streamline(ax, hs, opt, ko, a_grid, e_grid, cmap='plasma', **kwargs):
    """
    Plots the streamlines and their strength of the ode system in a and j space.
    a is the semimajor axis and j the specific angular momentum.
    The dissipative forces considered are to be given to opt

    Parameters:
        ax (plt.axes)                   : The axes on which to plot
        hs (merger_system.HostSystem)   : The object describing the properties of the host system
        opt (inspiral.Classic.EvolutionOptions) : The evolution options. The important part is opt.dissipativeForces
        ko  (KeplerOrbit)               : The parameters for a kepler orbit -- a and e will be ignored and taken from a_grid, e_grid
        a_grid (np.array)               : The grid in semimajor axis a
        e_grid (np.array)               : The grid in eccentricity e
        cmap (plt colormap) (optional)  : A plt colormap or the identifying name
        kwargs                          : Additional arguments passed to imshow, e.g. vmin/vmax

    Returns:
        out : matplotlib.image.AxesImage
            The axes image plotted
    """
    na = len(a_grid); ne = len(e_grid)

    a, e = np.meshgrid(a_grid, e_grid)
    da_grid = np.zeros(np.shape(a))
    de_grid = np.zeros(np.shape(e))

    # Calculate Derivatives
    for i in range(na):
        for k in range(ne):
            if a_grid[i]* (1.-e[k,i]) < 6.*hs.m1: # A bit smaller than the BH to have no alialising effects at the Loss Cone
                continue
            ko.a = a_grid[i]; ko.e = e[k,i]
            da_grid[k,i], dE_dt = inspiral.Classic.da_dt(hs, ko, opt=opt, return_dE_dt=True)
            de_grid[k,i] =  inspiral.Classic.de_dt(hs, ko, dE_dt=dE_dt, opt=opt)

    dloga = da_grid/a
    dlog1me = -de_grid/(1.-e)
    speed = np.sqrt(dlog1me**2 + dloga**2)

    # Plot Streamlines & Strength
    kwargs['norm'] = kwargs['norm'] if 'norm' in kwargs else 'log'
    im = ax.imshow(speed.T,
                   extent = [np.log10(1.-e_grid[0]), np.log10(1.-e_grid[-1]), np.log10(a_grid[0]/hs.r_isco), np.log10(a_grid[-1]/hs.r_isco)],
                  origin = 'lower', interpolation='bilinear', aspect='auto', cmap=cmap, zorder=1, **kwargs)
    strm = ax.streamplot(np.log10(1-e_grid), np.log10(a_grid/hs.r_isco),
                  dlog1me.T, dloga.T, color='black', zorder=1)
    # Cover Loss cone
    ax.plot(np.log10(1.-e_grid), np.log10(8./6./(1.-e_grid)), color='black', zorder=2)
    ax.fill_between(np.log10(1.-e_grid), np.log10(8./6./(1.-e_grid)), color='gray', alpha=1., zorder=2)
    return im

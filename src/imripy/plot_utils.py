import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from scipy.interpolate import interp1d

from imripy import halo, inspiral, waveform
from imripy import merger_system as ms

def plotEvolution(sp, ev, ax_a=None, ax_e=None, label="", ax_ae=None, ax_m=None, m2=1., ax_n=None, color=None, linestyle=None):
    """
    Plots the evolution of the system in the natural units that are used throughout the code.
    The evolution can be plotted as semimajor axis / time, eccentricity / time, eccentricity / semimajor axis, or relative mass/time,
        depending on which plt.axes objects are passed.
    The lines will have the same color and label

    Parameters:
        sp (merger_system.SystemProp)   : The object describing the properties of the inspiralling system
        ev (inspiral.Classic.Evolution) : The evolution object that results from the inspiral modeling
        ax_a (plt.axes)     (optional)  : The axes on which to plot semimajor axis / time
        ax_e (plt.axes)     (optional)  : The axes on which to plot eccentricity / time
        ax_ae (plt.axes)    (optional)  : The axes on which to plot eccentricity / semimajor axis
        ax_m (plt.axes)     (optional)  : The axes on which to plot relative mass / time
        m2   (float)        (optional)  : The initial mass of the system
        ax_n (plt.axes)     (optional)  : The axes on which to plot the braking index / frequency
        label (string)      (optional)  : The label corresponding to the lines
        **kwargs                        : Other parameters that can be passed to the plotting

    Returns:
        out : matplotlib.lines.Line2D
            The last line object plotted
    """
    if not ax_a is None:
        l, = ax_a.loglog(ev.t/ms.year_to_pc, ev.a/sp.r_isco(), label=label, color=color, linestyle=linestyle)
        color = l.get_c()
    if not ax_e is None:
        l, = ax_e.loglog(ev.t/ms.year_to_pc, ev.e, color=color, label=label, linestyle=linestyle)
        color = l.get_c()
    if not ax_m is None:
        l, = ax_m.loglog(ev.t/ms.year_to_pc, ev.m2/m2-1., color=color, label=label, linestyle=(linestyle if not linestyle is None else ':'))
        color = l.get_c()
    if not ax_ae is None:
        l, = ax_ae.plot(ev.a/sp.r_isco(), ev.e, color=color, label=label, linestyle=linestyle)
        color = l.get_c()
    if not ax_n is None:
        F, n = waveform.BrakingIndex(sp, ev)
        l, = ax_n.plot(2*F/ms.hz_to_invpc, n, color=color, label=label, linestyle=linestyle)
        color = l.get_c()
    return l

def plotGWcharacteristicStrain(sp, ev, ax_h, label="", acc=1e-13, harmonics=[2], color=None, **kwargs):
    """
    Plots the characteristic strain in its harmonics in the natural units that are used throughout the code.
    The lines will have the same color and alternating linestyle

    Parameters:
        sp (merger_system.SystemProp)   : The object describing the properties of the inspiralling system
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
    c = color if not color is None else next(color_cycle)['color']
    for n in harmonics:
        wf = waveform.h_n(n, sp, ev, acc=acc)
        l, = ax_h.loglog(wf[0]/ms.hz_to_invpc, 2.*wf[0]*np.abs(wf[1]), linestyle=next(linecycler),
                                 label=r"$h^{(" + str(n) +")}_{c,+," + label +"}$", color=c, **kwargs)


def plotDeltaN(sp_0, ev_0, sp_1, ev_1, ax_dN, ax_di=None, n=2, acc=1e-13, plotFgw5year=False, min_dN=10., **kwargs):
    """
    Plots the dephasing of a given harmonic in the natural units that are used throughout the code.

    Parameters:
        sp_0 (merger_system.SystemProp)   : The object describing the properties of the baseline inspiralling system
        ev_0 (inspiral.Classic.Evolution) : The evolution object that results from the baseline inspiral modeling
        sp_1 (merger_system.SystemProp)   : The object describing the properties of the alternative inspiralling system
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
    f_gw0, N_0 = waveform.N_cycles_n(n, sp_0, ev_0, acc=acc)
    N_0interp = interp1d(f_gw0, N_0, kind='cubic', bounds_error=False, fill_value=(0.,0.))
    f_gw1, N_1 = waveform.N_cycles_n(n, sp_1, ev_1, acc=acc)

    dN = N_1 - N_0interp(f_gw1)
    l, = ax_dN.loglog(f_gw1/ms.hz_to_invpc, np.abs(dN), **kwargs)

    if plotFgw5year:
        f_gw5yr = interp1d(ev_1.t, f_gw1, kind='cubic', bounds_error=True)(ev_1.t[-1] - 5.*ms.year_to_pc)
        ax_dN.axvline(f_gw5yr/ms.hz_to_invpc, linestyle='--', color=l.get_c())

    if not ax_di is None:
        ddN_df = np.gradient(dN, f_gw1)
        stop = np.where(np.abs(dN) < min_dN)[0]
        stop = stop[0] if len(stop) > 0 else len(ddN_df)
        if 'color' in kwargs:
            del kwargs['color']
        ax_di.plot(f_gw1[:stop]/ms.hz_to_invpc, (ddN_df/dN * f_gw1)[:stop], color=l.get_c(), **kwargs)

    return f_gw1, dN




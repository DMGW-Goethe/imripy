import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from scipy.interpolate import interp1d

from imripy import halo, inspiral, waveform
from imripy import merger_system as ms

def plotEvolution(sp, ev, ax_a=None, ax_e=None, label="", ax_ae=None, ax_m=None, m2=1.):
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
        label (string)      (optional)  : The label corresponding to the lines

    Returns:
        out : matplotlib.lines.Line2D
            The last line object plotted
    """
    l = None
    if not ax_a is None:
        l, = ax_a.loglog(ev.t/ms.year_to_pc, ev.a/sp.r_isco(), label=label)
    if not ax_e is None:
        l, = ax_e.loglog(ev.t/ms.year_to_pc, ev.e, color=(l.get_c() if not l is None else None), label=label)
    if not ax_m is None:
        l, = ax_m.loglog(ev.t/ms.year_to_pc, ev.m2/m2-1., linestyle='--', color=(l.get_c() if not l is None else None), label=label)
    if not ax_ae is None:
        l, = ax_ae.plot(ev.a/sp.r_isco(), ev.e, color=(l.get_c() if not l is None else None), label=label)
    return l

def plotGWcharacteristicStrain(sp, ev, ax_h, label="", acc=1e-13, harmonics=[2], color=None):
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
    """
    linecycler = cycle(["-", "--", "-.", ":"])
    color_cycle = ax_h._get_lines.prop_cycler
    l = None
    c = color if not color is None else next(color_cycle)['color']
    for n in harmonics:
        wf = waveform.h_n(n, sp, ev, acc=acc)
        l, = ax_h.loglog(wf[0]/ms.hz_to_invpc, 2.*wf[0]*np.abs(wf[1]), linestyle=next(linecycler), color=color,
                                 label=r"$h^{(" + str(n) +")}_{c,+," + label +"}$")


def plotDeltaN(sp_0, ev_0, sp_1, ev_1, ax_dN, n=2, label="",  acc=1e-13, color=None, linestyle=None):
    """
    Plots the dephasing of a given harmonic in the natural units that are used throughout the code.

    Parameters:
        sp_0 (merger_system.SystemProp)   : The object describing the properties of the baseline inspiralling system
        ev_0 (inspiral.Classic.Evolution) : The evolution object that results from the baseline inspiral modeling
        sp_1 (merger_system.SystemProp)   : The object describing the properties of the alternative inspiralling system
        ev_1 (inspiral.Classic.Evolution) : The evolution object that results from the alternative inspiral modeling
        ax_dN (plt.axes)                  : The axes on which to plot the difference in cycles left to observe
        n (int)             (optional)    : The harmonic to be plotted
        label (string)      (optional)   : The label corresponding to the lines
        color               (optional)   : The color of the lines. Can be anything that matplotlib accepts as color
        linestyle           (optional)   : The linestyle of the lines

    Returns:
        f_gw1 : array_like
            The frequencies corresponding to the difference in cycles
        dN   : array_like
            The difference in cycles of the two systems
    """
    f_gw0, N_0 = waveform.N_cycles_n(n, sp_0, ev_0, acc=acc)
    N_0interp = interp1d(f_gw0, N_0, kind='cubic', bounds_error=False, fill_value=(0.,0.))
    f_gw1, N_1 = waveform.N_cycles_n(n, sp_1, ev_1, acc=acc)

    dN = np.abs(N_1 - N_0interp(f_gw1))
    ax_dN.loglog(f_gw1/ms.hz_to_invpc, dN, color=color, linestyle=linestyle, label=label)
    return f_gw1, dN




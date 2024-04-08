import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

from scipy.interpolate import interp1d
from imripy import merger_system as ms, constants as c, halo, inspiral, kepler, plot_utils as pu
from collections.abc import Sequence

issequence = lambda x: isinstance(x, (Sequence, np.ndarray))

def assemble_figure(fig, axes_type):
    """
    This function assembles a figure object that can be used for making the animation. 
    Adds a gridspec layout with the 3d plot on the left and 1-4 plots on the right.
    These are saved in a dictionary for later use in animate().
    Supported axes are ['a', '1mea', 'pa', 'ia'], compare to plot_utils.plotEvolution
    This is optional and not perfect, adjust to your needs.
    
    Parameters
    -------
        fig : plt.Figure
            The figure object from matpotlib
        axes_type : list of axes types
            The names of the axes to be plotted, should be subset of ['a', '1mea', 'pa', 'ia']
        
    Returns
    -------
        axes : dict
            Dictionary with '3d'-> 3d axes, 'fig'-> figure object, and the axes types specified 
    """
    axes = {}
    n = len(axes_type)
    
    gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.55,0.45])
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
    axes['3d'] = fig.add_subplot(gs00[0,0], projection='3d')

    gs01 = gs0[1].subgridspec(1 if n < 2 else 2, 
                                1 if n <= 2 else 2, wspace=0.34, hspace=0.2)

    for i, ax in enumerate(axes_type):
        axes[ax] = fig.add_subplot(gs01[i % 2, int(i/2)])
        axes[ax].grid()
        if ax == 'a':
            axes[ax].set_xlabel("t / yrs")
            axes[ax].set_ylabel("a / $r_{isco}$")
        if ax == '1mea':
            axes[ax].set_xlabel("1-e")
            axes[ax].set_ylabel("a / $r_{isco}$")
        if ax == 'pa':
            axes[ax].set_xlabel("t / yrs")
            axes[ax].set_ylabel("$\omega$ / deg")
        if ax == 'ia':
            axes[ax].set_xlabel("t / yrs")
            axes[ax].set_ylabel("$\iota$ / $\deg$")
        

    axes['fig'] = fig
    return axes


def plot_orbit(hs, ax, ko, n=200, **kwargs):
    """ 
    This function plots the orbit on the 3d axes with a list of orbital points
    
    Parameters
    ---------
        hs : ms.HostSystem
        ax : plt.Axes
            The 3d axes
        ko : kepler.KeplerOrbit
            The orbit to be plotted
        n : int
            The number of points used in the plotting
        kwargs :
            Optional arguments passed to plt.plot, i.e. color, linestyle
    
    Returns
        l : matplotlib.lines.Line2D
            The line plotted
    """
    phi_list = np.linspace(0., 2.*np.pi, n)

    pos = np.array([ko.get_orbital_vectors(phi)[0] for phi in phi_list]) 
    pos /= hs.r_isco
    l, = ax.plot(pos[:,0], pos[:,1], pos[:,2], **kwargs)
    return l

def update_orbit(hs, l, ko, n=200):
    """ 
    This function updates the line object representing the orbit
    
    Parameters
    ---------
        hs : ms.HostSystem
        l : matplotlib.lines.Line2D
            The line of the orbit
        ko : kepler.KeplerOrbit
            The orbit to be plotted
        n : int
            The number of points used in the plotting
    """
    phi_list = np.linspace(0., 2.*np.pi, n)

    pos = np.array([ko.get_orbital_vectors(phi)[0] for phi in phi_list])
    pos /= hs.r_isco
    l.set_data_3d(pos[:,0], pos[:,1], pos[:,2])

def get_timestamps(hs, evs, t_real, fps=5., distribution='mixed_geometric', t_0=None, t_fin=None, matching=None):
    """
    This function tries to make sensible timesteps for the individual frames of the animation.
    There are different distributions of points available. 
    Also allows to match the points if multiple evolutions are provided.
    
    Parameters
    ---------
        hs : ms.HostSystem
            HostSystem object
        evs : single or list of inspiral.Classic.EvolutionObject
            The evolution object(s) 
        t_real : float
            The real time value the animation is supposed to be
        fps : float
            The number of frames per second
        distribution : one of ['linear', 'geometric', 'freq_linear', 'freq_geometric' 'mixed_geometric']
            The timesteps of the evolution are selected according to one of these schemes.
            linear/geometric is linear/geometric in time, with prefix freq this is linear/geometric in frequency
            mixed_geometric takes half of points geometric in time and half geomtric in frequency
        t_0 : float
            Samples start at t_0 
        t_fin : float
            Samples end at t_fin
        matching: 'freq' or 'time'
            If multiple evolutions are passed, their timesteps can be matched so that they are at the time/frequency
            in their evolutions. The first object is the reference point here.
        
    Returns
    -------
        t_steps : np.ndarray
            Array representing the times of the evolution objects that are used for the individual frames
            Shape is (len(evs), t_real*fps)
    """
    if not issequence(evs):
        evs = [evs]
    n =  int(t_real*fps)
    
    t_steps = np.zeros(shape=(len(evs), int(n)))
    
    for i, ev in enumerate(evs):
        left = t_0 or ev.t[0]
        right = t_fin or ev.t[-1]
        
        ev.f = np.sqrt(hs.m1/ev.a**3) / 2. / np.pi
        ev.t_to_f = interp1d(ev.t, ev.f, kind='cubic')#, bounds_error=False, fill_value=(0,0))
        ev.f_to_t = interp1d(ev.f, ev.t, kind='cubic')#, bounds_error=False, fill_value=(0,0))
        
        #print(i, left, right)
        if distribution == 'linear':
            t_steps[i,:] = np.linspace(left, right , n)
        elif distribution == 'geometric':
            t_steps[i,:] = np.geomspace(np.max([left, 1e-10]), right, n)
        elif distribution == 'freq_linear':
            t_steps[i,:] = ev.f_to_t(np.linspace( ev.t_to_f(left), ev.t_to_f(right), n))
        elif distribution == 'freq_geometric':
            t_steps[i,:] = ev.f_to_t(np.geomspace( ev.t_to_f(left), ev.t_to_f(right), n))
        elif distribution == 'mixed_geometric':            
            t_steps[i,:] = np.sort(np.append( np.geomspace(left if left > 0. else np.max([ev.t[1]/1e5, ev.options.accuracy or 1e-10]), right, int(n/2)),
                                                    ev.f_to_t(np.geomspace( ev.t_to_f(left), ev.t_to_f(right), (int(n/2)+1 if (n%2)==1 else int(n/2)) ) 
                                                 ))) 
        else:
            raise ValueError(f"Unrecognized distribution: {distribution}")
        
    if matching and matching == 'freq':
        for i, ev in enumerate(evs[1:]):
            t_steps[i+1,:] = ev.f_to_t( evs[0].t_to_f(t_steps[0,:]) )
    elif matching == 'time':
        for i, ev in enumerate(evs[1:]):
            t_steps[i+1,:] = t_steps[0,:]
        
    return t_steps

def handle_args(hs, evs, axes, t_steps, fps, colors, labels):
    if not issequence(evs):
        evs = [evs]
    
    if not '3d' in axes:
        raise ValueError("axes dict does not contain 3d axis")
        
    if not 'fig' in axes:
        raise ValueError("axes dict does not contain figure object")
    
    if not len(evs) == np.shape(t_steps)[0]:
        raise ValueError("t_steps shape {np.shape(t_steps)[0]} does not correspond to len(evs) = {len(evs)}")
        
    if not issequence(colors):
        colors = [colors]
    if not issequence(labels):
        labels = [labels]
    
    return hs, evs, axes, t_steps, fps, colors, labels

def animate(hs, evs, axes, t_steps, fps=5., colors=None, labels=None, plot_risco=True, **kwargs):
    """ 
    This function animates the inspiral(s) passed and returns the animation object.
    
    Parameters
    ---------
        hs : ms.HostSystem
            HostSystem object
        evs : single or list of inspiral.Classic.EvolutionResults
            The evolution object(s) to be animated
        axes : dict of plt.Axes objects
            A dictionary containing the axes to be populated
            Requires '3d'-> A 3d axes, and 'fig'->plt.figure object
            The other available axes types are 'a', '1mea', 'pa', 'ia'
            Compare to plot_utils.plotEvolution and assemble_figure 
        t_steps : np.ndarray
            An array with shape (len(evs), n) where n is the number of frames to be animated
            The array contains the timesteps of the evolutions to be plotted
            Can be created with get_timesteps
        fps : float
            The number of frames per second
        colors : single or list of matplotlib colors
            The colors for the evs
        labels : single or list of labels
            The labels for the evs
        plot_risco : Boolean
            Whether to plot the r_isco as a dashed line
        kwargs : Optional arguments passed to plotEvolution for the `background` lines
        
    Returns
    --------
        ani : matplotlib.animation.FuncAnimation
            The resultin animation
    """
    # handle args
    colors = colors or []
    labels = labels or []
    hs, evs, axes, t_steps, fps, colors, labels = handle_args(hs, evs, axes, t_steps, fps, colors, labels)
    
    # Plot Background
    axes['3d'].plot(0,0, marker='o', color='black') # MBH
    plot_orbit(hs, axes['3d'], kepler.KeplerOrbit(hs, 0., hs.r_isco), color='black', linestyle='--')
    for i, ev in enumerate(evs):
        l = pu.plotEvolution(hs, ev, ax_a=axes['a'] if 'a' in axes else None, 
                                     ax_1mea=axes['1mea'] if '1mea' in axes else None,
                                     ax_pa = axes['pa'] if 'pa' in axes else None,
                                     ax_ia = axes['ia'] if 'ia' in axes else None,
                                     color=None if len(colors) <= i else colors[i],
                                    label=labels[i] if len(labels) > i else None, **kwargs)
        if len(colors) <= i:
            colors.append(l.get_c())
    
    if '1mea' in axes:
        e_grid = 1.-np.array(axes['1mea'].get_xlim())
        axes['1mea'].plot(1.-e_grid, 8./6./(1.-e_grid), linestyle='--', color='black')

    
    # Plot Animated
    lines = {};
    for lbl, ax in axes.items():
        lines[lbl] = []
    a_curr = []
    
    for i, ev in enumerate(evs):
        t_0 = t_steps[i][0]
        ko = ev.get_kepler_orbit(t_0, interpolate=True); a_curr.append(ko.a)
        for lbl, ax in axes.items():
            if lbl == '3d':
                l = plot_orbit(hs, ax, ko, color=colors[i], label=labels[i] if len(labels) > i else None)
                lines['3d'].append(l)
            if lbl=='a':
                l, = ax.plot(t_0/c.year_to_pc if t_0 > 0. else ax.get_xlim()[0], ko.a/hs.r_isco, marker='o', 
                                     color=colors[i])
                lines['a'].append(l)
            if lbl=='1mea':
                l, = ax.plot(1.-ko.e, ko.a/hs.r_isco, marker='o', color=colors[i])
                lines['1mea'].append(l)
            if lbl=='pa':
                l, = ax.plot(t_0/c.year_to_pc, ko.periapse_angle*c.rad_to_arcmin, marker='o', color=colors[i])
                lines['pa'].append(l)
            if lbl=='ia':
                l, = ax.plot(t_0/c.year_to_pc, ko.inclination_angle*c.rad_to_arcmin, marker='o', color=colors[i])
                lines['ia'].append(l)
    
    a_max = np.max(a_curr)
    axes['3d'].set_xlim(-a_max/hs.r_isco, a_max/hs.r_isco)
    axes['3d'].set_ylim(-a_max/hs.r_isco, a_max/hs.r_isco)
    axes['3d'].legend()

    n = np.shape(t_steps)[1]
    def update_frame(frame):
        #print(f"{frame}/{n}")
        for i, ev in enumerate(evs):
            t_curr = t_steps[i, frame]
            if t_curr > ev.t[-1]:
                continue
            ko = ev.get_kepler_orbit(t_curr, interpolate=True)
            a_curr[i] = ko.a
            
            for lbl, ls in lines.items():
                if lbl == '3d':
                    update_orbit(hs, ls[i], ko)
                elif lbl=='a':
                    ls[i].set_data(t_curr/c.year_to_pc, ko.a/hs.r_isco)
                elif lbl=='1mea':
                    ls[i].set_data(1.-ko.e, ko.a/hs.r_isco)
                elif lbl=='pa':
                    ls[i].set_data(t_curr/c.year_to_pc, ko.periapse_angle*c.rad_to_arcmin)
                elif lbl=='ia':
                    ls[i].set_data(t_curr/c.year_to_pc, ko.inclination_angle*c.rad_to_arcmin)

        # update 3d axes?
        a_max = np.max(a_curr)
        xlim = axes['3d'].get_xlim()
        if a_max/hs.r_isco > xlim[1]:
            axes['3d'].set_xlim(-a_max/hs.r_isco, a_max/hs.r_isco)
            axes['3d'].set_ylim(-a_max/hs.r_isco, a_max/hs.r_isco)
        elif a_max/hs.r_isco < xlim[1]/10.:
            axes['3d'].set_xlim(-a_max/hs.r_isco, a_max/hs.r_isco)
            axes['3d'].set_ylim(-a_max/hs.r_isco, a_max/hs.r_isco)

    return FuncAnimation(axes['fig'], update_frame, frames=n, interval=1e3/fps)


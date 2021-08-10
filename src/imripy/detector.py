import numpy as np
from scipy.integrate import quad
import imripy.merger_system as ms


class Detector:
    """
    An abstract base class defining the basic properties of a gravitational wave detector

    """
    def Bandwith(self):
        """
        The bandwith of the detector in c/pc

        Returns:
            out : tuple
                The left and right boundary of the bandwith
        """
        return (-np.inf, np.inf)

    def NoiseSpectralDensity(self, f):
        """
        The noise spectral density of the detector at frequency f
            for an overview see https://arxiv.org/pdf/1408.0740.pdf

        Parameters:
            f : float or array_like
                The frequencies at which to evaluate the noise spectral density

        Returns:
            out : float
                The noise spectral density
        """
        pass

    def NoiseStrain(self, f):
        """
        The noise strain of the detector that needs to be compared to the characteristic strain of any signal
        Parameters:
            f : float or array_lik
                The frequencies at which to evaluate the noise strain

        Return:
            out: float or array_like
                The noise strain
        """
        return np.sqrt(f*self.NoiseSpectralDensity(f))

class eLisa(Detector):
    """
    A class describing the properties of eLisa
    """

    def Bandwith(self):
        """
        The bandwith of eLisa in c/pc

        Returns:
            out : tuple
                The left and right boundary of the bandwith
        """
        return (1e-4 * ms.hz_to_invpc, 1. * ms.hz_to_invpc)

    def NoiseSpectralDensity(self, f):
        """
        The noise spectral density of eLisa at frequency f in c/pc
            according to https://arxiv.org/pdf/1408.3534.pdf

        Parameters:
            f : float or array_like
                The frequencies at which to evaluate the noise spectral density in c/pc

        Returns:
            out : tuple
                The noise spectral density
        """
        S_acc = 2.13e-29 * (1. + 1e-4*ms.hz_to_invpc /f) * ms.m_to_pc**2 * ms.hz_to_invpc**3
        S_sn  = 5.25e-23 * ms.m_to_pc**2 * ms.s_to_pc
        S_omn = 6.28e-23 * ms.m_to_pc**2 * ms.s_to_pc
        l = 1e9 * ms.m_to_pc
        return 20./3. * (4.*S_acc/(2.*np.pi*f)**4 + S_sn + S_omn)/l**2  * (1. + (2.*f*l/0.41)**2)

class Lisa(Detector):
    """
    A class describing the properties of Lisa

    """
    def Bandwith(self):
        """
        The bandwith of Lisa in c/pc

        Returns:
            out : tuple
                The left and right boundary of the bandwith
        """
        return (1e-4 * ms.hz_to_invpc, 1. * ms.hz_to_invpc)

    def NoiseSpectralDensity(self, f):
        """
        The noise spectral density of Lisa at frequency f in c/pc
            as given in Eq.(13) of arxiv.org/pdf/1803.01944.pdf

        Parameters:
            f : float or array_like
                The frequencies at which to evaluate the noise spectral density in c/pc

        Returns:
            out : float or array_like
                The noise spectral density
        """
        P_oms = (1.5e-11 * ms.m_to_pc)**2  * (1. + (2e-3 * ms.hz_to_invpc/f)**4) / ms.hz_to_invpc
        P_acc = (3e-15 * ms.m_to_pc / ms.s_to_pc**2)**2 * (1. + (0.4e-3 * ms.hz_to_invpc/f)**2) * (1. + (f/8e-3/ms.hz_to_invpc)**4) / ms.hz_to_invpc
        f_s = 19.09e-3 * ms.hz_to_invpc
        L = 2.5e9 * ms.m_to_pc

        return 10./3./L**2  * (P_oms + 2.* (1. + np.cos(f/f_s)**2 ) * P_acc / (2.*np.pi*f)**4) * (1. + 6./10. * (f/f_s)**2)


def SignalToNoise(f, htilde, detector, acc=1e-13):
    """
    This function calculates the signal to noise ratio of a gravitational wave signal observed by a detector

    Parameters:
        f (array_like)       : The grid of frequencies
        htilde  (array_like) : The magnitude of the gravitational waveform in fourier space on the grid in frequencies
        detector (Detector)  : The object describing the detector properties
        acc  (float)         : An accuracy parameter that is passed to the integration method

    Returns:
        SoN : np.ndarray
            The signal to noise ratio over the grid of frequencies

    """
    f_min = np.max([f[0], detector.Bandwith()[0]])
    f_max = np.min([f[-1], detector.Bandwith()[1]])

    integrand = lambda f: htilde(f)**2 / detector.NoiseSpectralDensity(f) if f > f_min and f < f_max else 0.

    SoN = 4.*np.cumsum([quad(integrand, f[i-1], f[i], epsabs=acc, epsrel=acc, limit=200)[0] if not i == 0 else 0. for i in range(len(f))])
    SoN = np.sqrt(SoN)

    ''' # A different way of integrating
    integrand = lambda f: (f*htilde(f))**2 / detector.NoiseStrain(f)**2 if f > f_min and f < f_max else 0.
    SoN2 = 4.*np.cumsum([quad(lambda logf: integrand(np.exp(logf)), np.log(f[i-1]), np.log(f[i]), epsabs=acc, epsrel=acc, limit=200)[0] if not i == 0 else 0. for i in range(len(f))])
    SoN2 = np.sqrt(SoN2)

    print(SoN, SoN2)
    '''
    return SoN


## IMRIpy

This code allows the simulation of an Intermediate Mass Ratio Inspiral (IMRI), where a small compact object -- the secondary -- is around an orbit of a larger body -- the primary, usually a massive black hole (MBH). \
The system loses energy and angular momentum due to dissipative forces and slowly inspirals. The dissipative forces include gravitation wave (GW) emission, dynamical friction with dark matter (DM), interaction with an accretion disk, or stellar diffusion in stellar cusps.

The main purpose of the code is to solve the differential equations that arise in the modeling of IMRIs. The code can track the semimajor-axis, eccentricity, periapse angle, and inclination angle of the Kepler orbit. This orbit changes over time due to different dissipative forces. The code is very modular in design such that different dissipative forces can be added easily and their effects studied in detail. Additionally, the code can explore the phase space flows of the forces, compute the GW signal and compare it to LISA sensitivity.

The code also includes a stochastic description of some of the interactions with the help of stochastic differential equations (SDEs). This can be used to calculate rates and population behavior.

#### Models
For most of the functionality look at the dissertation [here](https://arxiv.org/pdf/2404.02808.pdf).

This code has been used in our publications [2112.09586](https://arxiv.org/abs/2112.09586) and [2211.05145](https://arxiv.org/pdf/2211.05145.pdf). See [this](https://github.com/DMGW-Goethe/imripy/blob/main/examples/2112.09586.ipynb) and [this](https://github.com/DMGW-Goethe/imripy/blob/main/examples/2211.05145.ipynb) file respectively for plot generation.

The code includes inspiral models from  \
[9402014](https://arxiv.org/abs/gr-qc/9402014) - gravitational wave emission \
[1408.3534](https://arxiv.org/abs/1408.3534.pdf), [2204.12508](https://arxiv.org/abs/2204.12508), [2305.17281](https://arxiv.org/pdf/2305.17281.pdf), [1711.09706](https://arxiv.org/abs/1711.09706.pdf) - DM halos + dynamical friction + accretion \
[1908.10241](https://arxiv.org/abs/1908.10241.pdf), [2107.00741](https://arxiv.org/abs/2107.00741.pdf), [1807.07163](https://arxiv.org/abs/1807.07163.pdf) - Keplerian orbits and waveforms \
[2002.12811](https://arxiv.org/abs/2002.12811.pdf),[2108.04154](https://arxiv.org/abs/2108.04154) - Halo Feedback \
[2207.10086](https://arxiv.org/abs/2207.10086), [10.1086/324713](https://iopscience.iop.org/article/10.1086/324713), [10.1093/mnras/stac1294](https://academic.oup.com/mnras/article/513/4/5465/6584408), [2205.10382](https://arxiv.org/pdf/2205.10382.pdf) - Accretion disk profile + interactions \
[1508.01390](https://arxiv.org/pdf/1508.01390.pdf) - Stellar Diffusion



#### Usage
See examples and dissertation folder.


### Install
Clone the repository and run \
__pip install -e__ . \
(the [-e option](https://pip.pypa.io/en/stable/cli/pip_install/#install-editable) allows you to continuously edit the files without recompiling, don't use if you don't need to edit the files)

If you would like to use the Stochastic module, download the [torchsde fork](https://github.com/niklasrb/torchsde) and install it.

### Citation
If you use this in your publications, please cite the two papers [2112.09586](https://inspirehep.net/literature/1993072), and [2211.05145](https://inspirehep.net/literature/2180435), and the [ASCL entry](https://ascl.net/2307.018).

#### License
See LICENSE File


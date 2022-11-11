## IMRIpy

This code allows the simulation of an Intermediate Mass Ratio Inspiral (IMRI) by gravitational wave emission with a Dark Matter(DM) halo or a (baryonic) Accretion Disk around the central Intermediate Mass Black Hole(IMBH).
It allows to use different density profiles (such as DM spikes), and different interactions, such as dynamical friction with and without HaloFeedback models or accretion.

#### Models
This code has been used in our publications [2112.09586](https://arxiv.org/abs/2112.09586) and [2211,05145]([https://arxiv.org/pdf/2211.05145.pdf). See [this](https://github.com/DMGW-Goethe/imripy/blob/main/examples/circularizationDynamicalFrictionPaper.ipynb) and [this](https://github.com/DMGW-Goethe/imripy/blob/main/examples/AccretionDiskvsDarkMatter.ipynb) file respectively for plot generation. \
The code includes inspiral models from  \
[9402014](https://arxiv.org/abs/gr-qc/9402014) - gravitational wave emission \
[1408.3534](https://arxiv.org/abs/1408.3534.pdf) - dynamical friction \
[1711.09706](https://arxiv.org/abs/1711.09706.pdf) - accretion \
[1908.10241](https://arxiv.org/abs/1908.10241.pdf), [2107.00741](https://arxiv.org/abs/2107.00741.pdf), [1807.07163](https://arxiv.org/abs/1807.07163.pdf) - Keplerian orbits \
[2002.12811](https://arxiv.org/abs/2002.12811.pdf),[2108.04154](https://arxiv.org/abs/2108.04154) - Halo Feedback \
[2207.10086](https://arxiv.org/abs/2207.10086), [10.1086/324713](https://iopscience.iop.org/article/10.1086/324713), [10.1093/mnras/stac1294](https://academic.oup.com/mnras/article/513/4/5465/6584408) - Accretion Disk profile & interactions


#### Usage
See examples


### Install
Clone the repository and run \
__pip install -e__ . \
(the [-e](https://pip.pypa.io/en/stable/cli/pip_install/#install-editable) option allows you to continuously edit the files without recompiling, don't use if you don't need to edit the files) \


#### License
See LICENSE File


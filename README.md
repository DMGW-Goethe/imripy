## IMRIpy

This code allows the simulation of an Intermediate Mass Ratio Inspiral (IMRI) by gravitational wave emission with a Dark Matter(DM) halo around the central Intermediate Mass Black Hole(IMBH).
It allows to use different halo models (such as DM spikes), and dynamical friction with and without HaloFeedback models.

#### Models
This code has been used in our publication [2112.09586](https://arxiv.org/abs/2112.09586). See [this](https://github.com/DMGW-Goethe/imripy/blob/main/examples/circularizationDynamicalFrictionPaper.ipynb) file for plot generation. \
The code includes inspiral models from  \
[9402014](https://arxiv.org/abs/gr-qc/9402014) - gravitational wave emission \
[1408.3534](https://arxiv.org/abs/1408.3534.pdf) - dynamical friction \
[1711.09706](https://arxiv.org/abs/1711.09706.pdf) - accretion \
[1908.10241](https://arxiv.org/abs/1908.10241.pdf), [2107.00741](https://arxiv.org/abs/2107.00741.pdf), [1807.07163](https://arxiv.org/abs/1807.07163.pdf) - Keplerian orbits \
[2002.12811](https://arxiv.org/abs/2002.12811.pdf),[2108.04154](https://arxiv.org/abs/2108.04154) - Halo Feedback \


#### Usage
See examples

#### TODO
 - Implement spike calculation from [1305.2619](https://arxiv.org/abs/1305.2619.pdf)


### Install
Clone the repository and run \
__pip install . -e__  \
(the [-e](https://pip.pypa.io/en/stable/cli/pip_install/#install-editable) option allows you to continuously edit the files without recompiling, don't use if you don't need to edit the files) \


#### License
See LICENSE File


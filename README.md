
[![Build/Test](https://github.com/mbruno46/pyobs/workflows/Build/Test/badge.svg)](https://github.com/mbruno46/pyobs/actions?query=workflow%3ABuild%2FTest)
![Build Doc](https://github.com/mbruno46/pyobs/workflows/Build%20Doc/badge.svg)
[![codecov](https://codecov.io/gh/mbruno46/pyobs/branch/master/graph/badge.svg)](https://codecov.io/gh/mbruno46/pyobs)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

# pyobs <img src="/doc/pyobs-logo.png" width="15%">

A Python library to analyse data generated 
from (Monte Carlo) Markov chains.

- **Website:** https://mbruno46.github.io/pyobs/
- **Documentation:** https://mbruno46.github.io/pyobs/
- **Examples:** [tests](./tests), [tutorials](./doc/tutorials)
- **Source code:** https://github.com/mbruno46/pyobs/
- **Bug reports:** https://github.com/mbruno46/pyobs/issues

If you use this library in your publications please consider citing:

* U. Wolff, [Monte Carlo errors with less errors](https://inspirehep.net/literature/621085). *Comput.Phys.Commun.* 156 (2004) 143-153.
* S. Schaefer, R. Sommer, F. Virotta, [Critical slowing down and error analysis in lattice QCD simulations](https://inspirehep.net/literature/871175). *Nucl.Phys.B* 845 (2011) 93-119.
* M. Bruno, R. Sommer, In preparation.

### Authors

Copyright (C) 2020, Mattia Bruno

## Installation

To install the library directly in your local python distribution,
simply run

```bash
pip install git+https://github.com/mbruno46/pyobs.git@master#egg=pyobs
# or for upgrading
pip install -U git+https://github.com/mbruno46/pyobs.git@master#egg=pyobs
```

After installation, `pyobs` can be imported like any other package 

```python
import pyobs
help(pyobs.observable)
```

The library can also be installed from a local clone of
the repository in *developer mode*, as described in the 
documentation. Recompilation of the C++ extensions might 
be necessary in this case.

## Example

```python
import numpy
import pyobs

data = numpy.loadtxt('plaquette.dat')

plaq = pyobs.observable(description='the plaquette')
plaq.create('ensembleA',data)

# perform arbitrary operations
print(plaq, plaq**2)

logplaq = pyobs.log(plaq)
```

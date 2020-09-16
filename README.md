
[![Build/Test](https://github.com/mbruno46/pyobs/workflows/Build/Test/badge.svg)](https://github.com/mbruno46/pyobs/actions?query=workflow%3ABuild%2FTest)
[![codecov](https://codecov.io/gh/mbruno46/pyobs/branch/master/graph/badge.svg)](https://codecov.io/gh/mbruno46/pyobs)

<img src="/docs/_images/pyobs-logo.png" width="25%">

# pyobs

A Python library to analyse data generated 
from (Monte Carlo) Markov chains.

## Authors

Copyright (C) 2020, Mattia Bruno

## Installation

To install the library directly in the local python distribution,
simply run the following commands

```bash
pip install git+https://github.com/mbruno46/pyobs.git@master#egg=pyobs
# or for upgrading
pip install -U git+https://github.com/mbruno46/pyobs.git@master#egg=pyobs
```

After installation, `pyobs` can be imported like any other package 

```python
import pyobs
from pyobs import obs
help(pyobs)
```

The library can also be installed from a local clone of
the repository or in *developer mode*, as described in the 
documentation (see link below). Recompilation of the 
C++ extensions might be necessary in this case.

## Documentation

The documentation together with tutorials
can be accessed in [HTML][1] format or [PDF][2].

[1]: https://mbruno46.github.io/pyobs
[2]: ./doc/pyobs-doc.pdf

If you use this library in your publications please consider citing:

* U. Wolff, "Monte Carlo errors with less errors". Comput.Phys.Commun. 156 (2004) 143-153.
* F. Virotta, "Critical slowing down and error analysis of lattice QCD simulations." PhD thesis.
* S. Schaefer, R. Sommer, F. Virotta, "Critical slowing down and error analysis in lattice QCD simulations". Nucl.Phys.B 845 (2011) 93-119.
* M. Bruno, R. Sommer, In preparation.


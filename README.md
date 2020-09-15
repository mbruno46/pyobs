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
# for upgrading
pip install --update git+https://github.com/mbruno46/pyobs.git@master#egg=pyobs
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

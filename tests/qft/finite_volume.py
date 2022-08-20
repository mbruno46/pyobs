import numpy
from scipy.integrate import quad

import pyobs

qc = pyobs.qft.finite_volume.quantization_condition_2to2
qcond2 = qc.single_channel(qc.com_frame(), 4, 1)

def tandelta(E):
    a = 0.220
    s = E**2
    return (1 - 4/s)**0.5 * a

qcond2.En(tandelta, 1)

en = qcond2.get_energy(tandelta)
for i in range(5):
    print(i, next(en))

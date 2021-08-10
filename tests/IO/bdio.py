import pyobs
import os

fname = f"{os.path.dirname(os.path.abspath(__file__))}/observable.bdio"

res = pyobs.load(fname)
[v, e] = pyobs.remove_tensor(res['observables'][0]).error()

assert abs(v-1.42252725) < 1e-8
assert abs(e-0.00566219) < 1e-8
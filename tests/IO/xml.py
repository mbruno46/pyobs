import pyobs
import os

fname = f"{os.path.dirname(os.path.abspath(__file__))}/observable.xml.gz"

res = pyobs.load(fname)
[v, e] = pyobs.remove_tensor(res).error()

assert abs(v-1.12207286) < 1e-8
assert abs(e-0.01189337) < 1e-8
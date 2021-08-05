import pyobs

res = pyobs.load('./observable.xml.gz')
[v, e] = pyobs.remove_tensor(res).error()

assert abs(v-1.12207286) < 1e-8
assert abs(e-0.01190638) < 1e-8
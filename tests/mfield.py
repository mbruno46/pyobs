import pyobs
import numpy
import gzip

L=[32,32,32]
data = numpy.loadtxt('./mfield-32x32x32-obs.txt.gz')

mfobs = pyobs.obs()
mfobs.create(f'test-mfield',data,lat=L)
mfobs.peek()

print(mfobs)

print('a random operation',mfobs * 45 + mfobs**2)

[v, e] = mfobs.error(errinfo={'test-mfield': pyobs.errinfo(R=12)})

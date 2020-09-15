import pyobs
import numpy
import os

L=[32,32,32]
p=os.path.realpath(__file__)[:-9]
data = numpy.loadtxt(f'{p}/mfield-32x32x32-obs.txt.gz')

mfobs = pyobs.obs()
mfobs.create(f'test-mfield',data,lat=L)
mfobs.peek()

print(mfobs)

print('a random operation',mfobs * 45 + mfobs**2)

[v, e] = mfobs.error(errinfo={'test-mfield': pyobs.errinfo(R=12)})

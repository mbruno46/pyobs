import pyobs
import numpy

arr = pyobs.import_string(['1234(4)','0.02345(456)'])
print(arr)

tex = pyobs.tex_table(numpy.c_[[1,2],arr,arr,arr],['d',0,0,'.2f','.4f',0,0])
for t in tex:
    print(t)
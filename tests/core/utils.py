import pyobs
import numpy

print(pyobs.import_string('1234(56)'))

input = ['1234(4)','0.02345(456)']
print(input)

arr = pyobs.import_string(input)
print(arr)

print(pyobs.valerr(1234,4,significant_digits=1))

print(pyobs.valerr(arr[:,0],arr[:,1]))

tex = pyobs.tex_table(numpy.c_[[1,2],arr,arr,arr],['d',0,0,'.2f','.4f',0,0])
for t in tex:
    print(t)
    

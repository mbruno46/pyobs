/********************************************************************************

 mftools.cc: C++ fast methods for the master-field error
 Copyright (C) 2020 Mattia Bruno
 
 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 
********************************************************************************/

#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL clib_ARRAY_API

#include "numpy/arrayobject.h"

#include <stdlib.h>
#include <math.h>
#include <complex>
#include <vector>
#include <string>


static PyObject* intrsq(PyObject *self, PyObject *args)
{
   int mu,ir,d,s,D,v,rrmax,*x,*lat;
   double *data, *sum;
   PyArrayObject *in, *out, *_lat;
   PyObject *_rrmax;

   if (!PyArg_ParseTuple(args, "O!O!O", &PyArray_Type, &in, &PyArray_Type, &_lat, &_rrmax)) {
      return NULL;
   }

   if ((PyArray_DESCR(_lat)->type_num!=NPY_INT32) || (PyArray_NDIM(_lat)!=1))
   {
      PyErr_SetString(PyExc_ValueError, "Unexpected lat");
      return NULL;
   }
   if ((PyArray_DESCR(in)->type_num!=NPY_FLOAT64) || (PyArray_NDIM(in)!=1))
   {
      PyErr_SetString(PyExc_ValueError, "Unexpected data");
      return NULL;
   }
   data = (double*)PyArray_DATA(in);
   lat = (int*)PyArray_DATA(_lat);
   rrmax = PyLong_AsLong(_rrmax);

   //D=(int)(PyList_Size(_lat));
   D=(int)(PyArray_DIMS(_lat)[0]);
   v=1;
   for (mu=0; mu<D; mu++) 
      v *= lat[mu];

   x = new int[D];
   for (mu=0;mu<D;mu++)
      x[mu]=0;

   npy_intp nn = rrmax;
   out = (PyArrayObject*)PyArray_SimpleNew(1, &nn, NPY_FLOAT64);
   sum = (double*)PyArray_DATA(out);
   memset(sum,0,sizeof(double)*rrmax);

   sum[0]=data[0];
   for (s=1;s<v;s++)
   {
      x[D-1]++;
      ir=0;
      for (mu=D-1;mu>=0;mu--)
      {
         if ((x[mu]==lat[mu]) && (mu>0))
         {
            x[mu-1]++;
            x[mu]=0;
         }
        
         d=x[mu];
         if (x[mu]>lat[mu]/2)
            d-=lat[mu];
         ir+=d*d;
      }
      if (ir<rrmax)
         sum[ir] += data[s];
   }

   delete x;
   //PyArray_CLEARFLAGS(out,NPY_ARRAY_WRITEABLE);
   return (PyObject*)out;
}


static PyObject* idx2rsq(PyObject *self, PyObject *args)
{
   int *lat,v,s,mu,ir,d,*x,D;
   int32_t *rsq;
   PyArrayObject *a;
   PyObject *_lat;

   if (!PyArg_ParseTuple(args, "O", &_lat)) {
      return NULL;
   }
   
   if (!PyList_Check(_lat)) {
      return NULL;
   }

   D=(int)(PyList_Size(_lat));
   v=1;
   lat=new int[D];
   for (size_t i=0; i<(size_t)D; i++) 
   {
      lat[i]=PyLong_AsLong(PyList_GetItem(_lat,i));
      v *= lat[i];
   }

   x = new int[D];
   for (mu=0;mu<D;mu++)
      x[mu]=0;
   npy_intp nn = v;
   a = (PyArrayObject*)PyArray_SimpleNew(1, &nn, NPY_INT32);
   rsq = (int32_t*)PyArray_DATA(a);

   rsq[0]=0;
   for (s=1;s<v;s++)
   {
      x[D-1]++;
      ir=0;
      for (mu=D-1;mu>0;mu--)
      {
         if (x[mu]==lat[mu])
         {
            x[mu-1]++;
            x[mu]=0;
         }
        
         d=x[mu];
         if (x[mu]>lat[mu]/2)
            d-=lat[mu];
         ir+=d*d;
      }
      rsq[s]=(int32_t)ir;
   }

   delete x;
   delete lat;

   PyArray_CLEARFLAGS(a,NPY_ARRAY_WRITEABLE);
   return (PyObject*)a;
}


static PyMethodDef methods[] = {
   {"idx2rsq",  idx2rsq, METH_VARARGS, "idx2rsq"},
   {"intrsq",  intrsq, METH_VARARGS, "intrsq"},
   {NULL, NULL, 0, NULL}
};


static struct PyModuleDef methods_def = {
   PyModuleDef_HEAD_INIT,
   "mftools",   /* name of module */
   "tools for mfield", /* module documentation */
   -1,       /* m_size */
   methods, /* methods */
   NULL,
   NULL,
   NULL,
   NULL,
};


PyMODINIT_FUNC PyInit_mftools(void)
{
   import_array();
   return PyModule_Create(&methods_def);
}

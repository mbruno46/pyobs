
#include <stdlib.h>
#include <math.h>

void compute_drho(double *drho, double *rho, int tmax, int N)
{
   int t,k;
   double h,hh;

   drho[0]=0.0;
   for (t=1;t<tmax;t++)
   {
      hh = 0.;
      
      for (k=1;k<tmax-t;k++)
      {
	 h = rho[k+t];
	 h += rho[abs(k-t)];
	 h -= 2.0*rho[t]*rho[k];
	 hh += h*h;
      }

      drho[t] = sqrt( hh / (double)(N) );
   }
}

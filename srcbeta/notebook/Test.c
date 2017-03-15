/******************************************************************************
 *                     Code generated with sympy 0.7.6.1                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                        This file is part of 'CRTBP'                        *
 ******************************************************************************/
#include "Test.h"
#include <math.h>

void EOM(double Isp, double T, double ax, double ay, double az, double g0, double lvx, double lvy, double lvz, double lx, double ly, double lz, double m, double mu, double u, double vx, double vy, double vz, double x, double y, double z, double *out_2239394031191926941) {

   out_2239394031191926941[0] = vx;
   out_2239394031191926941[1] = vy;
   out_2239394031191926941[2] = vz;
   out_2239394031191926941[3] = T*ax*u/m - mu*(mu + x - 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 3.0L/2.0L) + 2*vy + x - (-mu + 1)*(mu + x)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 3.0L/2.0L);
   out_2239394031191926941[4] = T*ay*u/m - mu*y/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 3.0L/2.0L) - 2*vx - y*(-mu + 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 3.0L/2.0L) + y;
   out_2239394031191926941[5] = T*az*u/m - mu*z/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 3.0L/2.0L) + z*(mu - 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 3.0L/2.0L);
   out_2239394031191926941[6] = -T*u/(Isp*g0);
   out_2239394031191926941[7] = -lvx*(-mu*(-3*mu - 3*x + 3)*(mu + x - 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 5.0L/2.0L) - mu/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 3.0L/2.0L) - (-3*mu - 3*x)*(-mu + 1)*(mu + x)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 5.0L/2.0L) - (-mu + 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 3.0L/2.0L) + 1) - lvy*(-mu*y*(-3*mu - 3*x + 3)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 5.0L/2.0L) - y*(-3*mu - 3*x)*(-mu + 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 5.0L/2.0L)) - lvz*(-mu*z*(-3*mu - 3*x + 3)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 5.0L/2.0L) + z*(-3*mu - 3*x)*(mu - 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 5.0L/2.0L));
   out_2239394031191926941[8] = -lvx*(3*mu*y*(mu + x - 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 5.0L/2.0L) + 3*y*(-mu + 1)*(mu + x)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 5.0L/2.0L)) - lvy*(3*mu*pow(y, 2)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 5.0L/2.0L) - mu/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 3.0L/2.0L) + 3*pow(y, 2)*(-mu + 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 5.0L/2.0L) - (-mu + 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 3.0L/2.0L) + 1) - lvz*(3*mu*y*z/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 5.0L/2.0L) - 3*y*z*(mu - 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 5.0L/2.0L));
   out_2239394031191926941[9] = -lvx*(3*mu*z*(mu + x - 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 5.0L/2.0L) + 3*z*(-mu + 1)*(mu + x)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 5.0L/2.0L)) - lvy*(3*mu*y*z/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 5.0L/2.0L) + 3*y*z*(-mu + 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 5.0L/2.0L)) - lvz*(3*mu*pow(z, 2)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 5.0L/2.0L) - mu/pow(pow(y, 2) + pow(z, 2) + pow(mu + x - 1, 2), 3.0L/2.0L) - 3*pow(z, 2)*(mu - 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 5.0L/2.0L) + (mu - 1)/pow(pow(y, 2) + pow(z, 2) + pow(mu + x, 2), 3.0L/2.0L));
   out_2239394031191926941[10] = 2*lvy - lx;
   out_2239394031191926941[11] = -2*lvx - ly;
   out_2239394031191926941[12] = -lz;
   out_2239394031191926941[13] = T*ax*lvx*u/pow(m, 2) + T*ay*lvy*u/pow(m, 2) + T*az*lvz*u/pow(m, 2);

}

import jax.numpy as jnp
import jax
from jax import random
from jaxopt import ProximalGradient, GradientDescent, BlockCoordinateDescent
from jaxopt.prox import prox_lasso
import pandas as pd
from jax import jit
from jaxopt import objective
from jaxopt import prox


#Limb-darkeing laws of g-, r-, i-, z-bands for EV Lac
ld_star = jnp.array([[ 0.0218107,   2.22674688, -1.9960983,0.64901019],[-0.11589031,  2.15751326, -1.73046154,  0.51514061],[0.66485334,0.94296271,-1.23621597, 0.46103881,], [ 1.09204044, -0.02303873, -0.48276016,0.23430949,]])
ld_spot = jnp.copy(ld_star)

#Number of color
num_color = ld_star.shape[0]

#The flux of Unspotted photosphere 
flux_lim = 1 - ld_star[0:num_color,0]/5. - ld_star[0:num_color,1]*2./6. - ld_star[0:num_color,2]*3./7. -ld_star[0:num_color,3]*4./8

#The number of grids for latitude and longitude
num_phi = 1801
num_lam = 3601
#Infinitesimal of latitude and longitude
dphi = jnp.pi/(num_phi-1)
dlam = jnp.pi*2./(num_lam-1)
#Grid for latitude and longitude

lat = jnp.linspace(-jnp.pi/2.,jnp.pi/2.,num_phi)

num_lam = num_lam -1 
lon = jnp.linspace(-jnp.pi,jnp.pi-dlam,num_lam)


#Rotation period
period = 4.359
#Inclination angle
incl = 60.*jnp.pi/180.



#x,y,z-coordinate by inclination, latitude, and longitude 
@jit
def xf(phi,lam):
    return jnp.dot(jnp.cos(phi).reshape(-1,1),jnp.sin(lam).reshape(1,-1))
@jit
def yf(phi,lam):
    return jnp.sin(incl)*jnp.dot(jnp.sin(phi).reshape(-1,1),jnp.ones([lam.size]).reshape(1,-1))-jnp.cos(incl)*jnp.dot(jnp.cos(phi).reshape(-1,1),jnp.cos(lam).reshape(1,-1))
@jit
def zf(phi,lam):
    return jnp.cos(incl)*jnp.dot(jnp.sin(phi).reshape(-1,1),jnp.ones([lam.size]).reshape(1,-1))+jnp.sin(incl)*jnp.dot(jnp.cos(phi).reshape(-1,1),jnp.cos(lam).reshape(1,-1))

#product of cos(phi) and cos(beta) (z-coordinate)
@jit
def cosphi_zf(phi,lam):
    return jnp.cos(incl)*jnp.dot((jnp.sin(phi)*jnp.cos(phi)).reshape(-1,1),jnp.ones([lam.size]).reshape(1,-1))+jnp.sin(incl)*jnp.dot((jnp.cos(phi)**2).reshape(-1,1),jnp.cos(lam).reshape(1,-1))





#Time-series of cos(beta)
@jit
def cos_beta(phi,lam):
    z1 = jnp.cos(incl)*jnp.dot(jnp.sin(phi).reshape(-1,1),jnp.ones([lam.size]).reshape(1,-1)) #(num_phi, num_lam)
    z1 = z1.reshape(1,z1.size) #(1, num_phi*num_lam)
    z1 = jnp.dot(jnp.ones([t.size]).reshape(-1,1),z1) #(num_t, num_phi*num_lam)
    z2 = jnp.sin(incl)*jnp.dot(jnp.cos(phi).reshape(-1,1),jnp.cos(lam).reshape(1,-1))
    z2 = z2.reshape(1,z2.size)
    z2 = jnp.dot(jnp.cos(2.*jnp.pi*t/period).reshape(-1,1),z2)
    z3 = jnp.sin(incl)*jnp.dot(jnp.cos(phi).reshape(-1,1),jnp.sin(lam).reshape(1,-1))
    z3 = z3.reshape(1,z3.size)
    z3 = jnp.dot(jnp.sin(2.*jnp.pi*t/period).reshape(-1,1),z3)
    return z1+z2-z3


#Time-series of cos(phi)*cos(beta)
@jit
def cos_phi_beta(phi,lam):
    z1 = jnp.cos(incl)*jnp.dot((0.5*jnp.sin(2.0*phi)).reshape(-1,1),jnp.ones([lam.size]).reshape(1,-1))
    z1 = z1.reshape(1,z1.size)
    z1 = jnp.dot(jnp.ones([t.size]).reshape(-1,1),z1)
    z2 = jnp.sin(incl)*jnp.dot((jnp.cos(phi)**2).reshape(-1,1),jnp.cos(lam).reshape(1,-1))
    z2 = z2.reshape(1,z2.size)
    z2 = jnp.dot(jnp.cos(2.*jnp.pi*t/period).reshape(-1,1),z2)
    z3 = jnp.sin(incl)*jnp.dot((jnp.cos(phi)**2).reshape(-1,1),jnp.sin(lam).reshape(1,-1))
    z3 = z3.reshape(1,z3.size)
    z3 = jnp.dot(jnp.sin(2.*jnp.pi*t/period).reshape(-1,1),z3)
    return z1+z2-z3



#Summation of flux
@jit
def flux(phi, lam, fc, band):
    #z = jnp.maximum(cos_beta(phi,lam),1.0e-12)
    cosb = cos_beta(phi,lam)
    z = jnp.where(cosb<0,1e-12,cosb)
    ld_law = (ld_star[band][0]-fc*ld_spot[band][0])*(1.-jnp.sqrt(z))+(ld_star[band][1]-fc*ld_spot[band][1])*(1.-z) 
    ld_law += (ld_star[band][2]-fc*ld_spot[band][2])*(1.-jnp.sqrt(z)*z)+(ld_star[band][3]-fc*ld_spot[band][3])*(1.-z*z)
    phi_cosb = cos_phi_beta(phi,lam)
    phi_z = jnp.where(phi_cosb<0,1e-12,phi_cosb)
    return jnp.sum(phi_z*((1.-jnp.tile(fc,(t.size)).reshape(t.size,fc.size))-ld_law),axis=1)


#Relative flux
@jit
def relative_flux(fc):
    f = jnp.zeros([num_color,t.size])
    index = jnp.searchsorted(rel_int[0], 1.-fc)
    for j in range(num_color):
        f = f.at[j].set(flux_lim[j] - dphi*dlam*flux(lat,lon,rel_int[j,index],j)/jnp.pi)
    f_ave = jnp.mean(f,axis=1)+1e-12
    f_ave = f_ave.reshape(-1,1)
    f = f/f_ave-1.
    return f

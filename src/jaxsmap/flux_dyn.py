import jax.numpy as jnp
import jax
from jax import random
from jax import jit

@jit
def flux_phot(ld):
    return 1. - ld[0]/5. - ld[1]*2./6. - ld[2]*3./7. -ld[3]*4./8

@jit
def local_area(phi,lam,incl):
    return jnp.cos(incl)*jnp.dot((jnp.sin(phi)*jnp.cos(phi)).reshape(-1,1),jnp.ones([lam.size]).reshape(1,-1))+jnp.sin(incl)*jnp.dot((jnp.cos(phi)**2).reshape(-1,1),jnp.cos(lam).reshape(1,-1))

    

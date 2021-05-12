import numpy as np
from scipy.special import hyp2f1


def dis(x, undo=False, z=0.0, dis_std=6.0, **kwargs):
    dis_norm = dis_std * D(z)  # [Mpc/h]

    if not undo:
        dis_norm = 1 / dis_norm

    x *= dis_norm

def vel(x, undo=False, z=0.0, dis_std=6.0, **kwargs):
    vel_norm = dis_std * D(z) * H(z) * f(z) / (1 + z)  # [km/s]

    if not undo:
        vel_norm = 1 / vel_norm

    x *= vel_norm


def D(z, Om=0.31):
    """linear growth function for flat LambdaCDM, normalized to 1 at redshift zero
    """
    OL = 1 - Om
    a = 1 / (1+z)
    return a * hyp2f1(1, 1/3, 11/6, - OL * a**3 / Om) \
             / hyp2f1(1, 1/3, 11/6, - OL / Om)

def f(z, Om=0.31):
    """linear growth rate for flat LambdaCDM
    """
    OL = 1 - Om
    a = 1 / (1+z)
    aa3 = OL * a**3 / Om
    return 1 - 6/11*aa3 * hyp2f1(2, 4/3, 17/6, -aa3) \
                        / hyp2f1(1, 1/3, 11/6, -aa3)

def H(z, Om=0.31):
    """Hubble in [h km/s/Mpc] for flat LambdaCDM
    """
    OL = 1 - Om
    a = 1 / (1+z)
    return 100 * np.sqrt(Om / a**3 + OL)

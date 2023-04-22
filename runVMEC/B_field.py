# ---------------------------------------------------------------

# -------- READ VMEC file to determine minor and major radii

# ---------------------------------------------------------------
from wout_read import readWout
import numpy as np
woutfile = 'wout_HSX_aux.nc'

# define theta and phi angle arrays to be evaluated

ntheta = 20

theta = np.linspace(-np.pi, np.pi, ntheta)

nphi = 720

phivmec = np.linspace(0, 2 * np.pi, nphi)

# open and read file


def B_field(job, inp):
    basePath = "/home/smiskey/Documents/HSX/VMEC" #Edit this VMEC PATH
    vmec_file_path = basePath + '/Jobs{}'.format(job) + "/" + inp
    wout = readWout(path=vmec_file_path, name=woutfile,
                    iotaPro=True, curvAmps=True, diffAmps=True)

    # evaluate R,Z and Bmod

    keys = ['R', 'Z', 'Bmod']

    wout.transForm_3D(theta, phivmec, keys)

    Bmod = wout.invFourAmps['Bmod']

    B_mean = np.mean(Bmod[1, :, :])
    print('mean magnetic field along the axis: %.3f' % B_mean, 'T')

    B_req = 1
    ratio = B_mean/B_req
    I_start = -10722.0
    I_dif = I_start*ratio - I_start
    I_new = I_start - I_dif

    print("New Current to make mean B field 1 T: ", I_new)

    return I_new

# -------------------

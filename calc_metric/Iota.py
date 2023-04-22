import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from wout_read import readWout
from scipy.optimize import curve_fit
woutfile = 'wout_HSX_aux_opt0.nc'

# define theta and phi angle arrays to be evaluated

ntheta = 20

theta = np.linspace(-np.pi, np.pi, ntheta)

nphi = 720

phivmec = np.linspace(0, 2 * np.pi, nphi)
job = "Predicted"  # Edit for different sets
os.system('mkdir Plots/set_{}'.format(job))
numDir = 65  # Edit for number of files in a set
states = np.empty((numDir, 1))

# define the true objective function


def objective(x, a, b, c, d, e, f):
    return a * x + b * x**2 + c * x**3 + d * x**4 + e * x**5 + f


for i in range(numDir):  # EDIT RANGE
    num = i
    vmec_file_path = '/media/smiskey/USB/main_coil_0/set_{}_data_files'.format(
        job) + '/job_{}'.format(num)  # Check This

    wout = readWout(path=vmec_file_path, name=woutfile,
                    iotaPro=True, curvAmps=True, diffAmps=True)

    matplotlib.rcParams.update({'font.size': 18})

    plt.close('all')

    rational_surface1 = 4/4.

    rational_surface2 = 8/7.

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    ax3.grid()

    ax3.plot(np.sqrt(wout.s_dom), np.abs(wout.iota),
             label='rotational transform', color='k')

    ax3.plot([0, 1.1], [rational_surface1, rational_surface1],
             color='tab:blue', linestyle='dashdot')

    ax3.plot([0, 1.1], [rational_surface2, rational_surface2],
             color='tab:orange', linestyle='dashdot')

    ax3.legend(fontsize=12)

    ax3.set_xlabel('r/a')

    ax3.set_ylabel(r'$\iota/2\pi$')

    fig3.tight_layout()

    index1 = np.argmin(np.abs(wout.iota-rational_surface1))
    ax3.plot(np.sqrt(wout.s_dom)[index1], np.abs(wout.iota)[
             index1], '*', markersize=13, color='tab:blue')

    minimum_iota_distance1 = np.abs(
        np.abs(wout.iota)[index1]-rational_surface1)

    index2 = np.argmin(np.abs(wout.iota-rational_surface2))

    ax3.plot(np.sqrt(wout.s_dom)[index2], np.abs(wout.iota)[
             index2], '*', markersize=13, color='tab:orange')

    minimum_iota_distance2 = np.abs(
        np.abs(wout.iota)[index2]-rational_surface2)

    def curve_fitter(width):
        x = np.sqrt(wout.s_dom)
        y = np.abs(wout.iota)
        popt, _ = curve_fit(objective, x, y)
        a, b, c, d, e, f = popt
        print('y = %.5f * x + %.5f * x^2 + %.5f * x^3 + %.5f * x^4 + %.5f * x^5 + %.5f' %
              (a, b, c, d, e, f))
        # define a sequence of inputs between the smallest and largest known inputs
        x_line = np.arange(min(x), width, 0.01)
        # calculate the output for the range
        y_line = objective(x_line, a, b, c, d, e, f)
        ax3.plot(x_line, y_line, '--', color='red')
        return x_line, y_line

    n1 = round(minimum_iota_distance1, 3)
    n2 = round(minimum_iota_distance2, 3)

    def heavy_side(n1, n2):
        if n1 == 0.000 or n2 == 0.000:
            delta_iota_H = 0
            return delta_iota_H
        # Heavy side function for iota below 4/4 rational surface
        if minimum_iota_distance1 < 0.1 and np.abs(wout.iota)[index1] < rational_surface1:
            # Sets the test allowable r/a width after 1 for the 4/4 rational surface and fits the curve polynomially
            y_line = curve_fitter(width=1.2)[1]
            if max(y_line) > rational_surface1:
                delta_iota_H = 0
                return delta_iota_H
            else:
                delta_iota_H = 1
                return delta_iota_H
        # Heavy side function for iota below 8/7 rational surface
        if minimum_iota_distance2 < 0.05 and np.abs(wout.iota)[index2] < rational_surface2:
            # Sets the test allowable r/a width after 1 for the 8/7 rational surface and fits the curve polynomially
            y_line = curve_fitter(width=1.1)[1]
            if max(y_line) > rational_surface2:
                delta_iota_H = 0
                return delta_iota_H
            else:
                delta_iota_H = 1
                return delta_iota_H
        # Heavy side functiuon for iota between 4/4 and 8/7 that does not cross either line.
        else:
            # Runs to get curve fit between 4/4 and 8/7
            curve_fitter(width=1.1)
            delta_iota_H = 1
            return delta_iota_H
    delta_iota_H = heavy_side(n1, n2)
    states[num-1] = np.array([delta_iota_H])
    fig3.savefig('Plots/set_{}/'.format(job) +
                 'rotational_transform{}.png'.format(num))  # Check This

output = open('distances.txt', 'w')
for i in states:
    output.write(' '.join(['%.3f' % (s) for s in i]) + '\n')
output.close()
# Check This
Iotaplots = "/home/smiskey/Documents/HSX/Scripts/Plots/set_{}".format(job)
command = "mv distances.txt " + Iotaplots
os.system(command)

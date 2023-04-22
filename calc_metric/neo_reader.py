import os
import numpy as np
import wout_reader as wr


class read_neo:
    """
    A class to read ascii neo files from STELLOPT

    ...

    Attributes
    ----------
    neo_path : str
        Full path to the neo output file.

    wout_path : str
        Full path to the wout output file.

    Methods
    -------
    average_epsilon(self)
        Calculate the radial average epsilon effective

    Raises
    ------
    IOError
        either the neo or wout output file does not exist
    """

    def __init__(self, neo_path, wout_path):

        if not os.path.isfile(neo_path):
            raise IOError("Path does not exist : "+neo_path)

        if not os.path.isfile(wout_path):
            raise IOError("Path does not exist : "+wout_path)

        wout = wr.read_wout(wout_path)
        s_dom = wout.s_grid

        eps_eff = np.empty(s_dom.shape)
        eps_eff[:] = np.nan

        with open(neo_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                idx = int(line[0])-1
                eps_eff[idx] = float(line[1])

        is_nan = np.isnan(eps_eff)
        not_nan = ~is_nan

        self.s_dom = s_dom[not_nan]
        self.eps_eff = eps_eff[not_nan]

    def average_epsilon(self):
        """
        Calculate the radial average epsilon effective
        """
        return np.trapz(self.eps_eff, self.s_dom) / (self.s_dom[-1] - self.s_dom[0])
    
    def plot_epsilon(self):
        return self.eps_eff, self.s_dom

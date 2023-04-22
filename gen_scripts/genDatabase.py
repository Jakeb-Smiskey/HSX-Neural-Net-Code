import os
import h5py
import numpy as np
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

inp = input('Please DataFile directory:' )
dump = input("Please input dump location:" )
base_dirc = os.path.join('/media/smiskey/USB/main_coil_0', inp) #Change set #
dump_path = os.path.join('/media/smiskey/USB/main_coil_0', dump, 'metric_database.txt')
w = open(dump_path, 'w')
w.write('Eps_eff QHS B_avg F_kappa G_ss kappa delta rho Q_seq0p5 s_hat\n')
dirlist = sorted_alphanumeric(os.listdir(base_dirc))
print(dirlist)

for i in dirlist:
	filepath = os.path.join(base_dirc,i, "metric_output.txt")
	f = open(filepath, "r")
	data = f.readlines()
	f.close()
	w.write(data[1])
w.close()



import os
import h5py
import numpy as np


inp = input("Please input set directory:")
dump_path = os.path.join('/media/smiskey/USB/main_coil_0',
                          inp, 'metric_database.txt')  # Change location
metric_qhs = os.path.join('/media/smiskey/USB/main_coil_0',
                          'metric_QHS_1T.txt')  # Change Location


number = 1.2 #SET THIS TO AUX COIL NUMBER
size = 65 #CHANGE THIS TO FIT SIZE OF DATABASE



qhs = np.loadtxt(metric_qhs, dtype=float, skiprows=1)
File_data = np.loadtxt(dump_path, dtype=float, skiprows=1)
File_names = np.loadtxt(dump_path, dtype=str, max_rows=1)
normalized = File_data/qhs
lst = []
n = 0
for i in normalized:
    if i[5] > 2:
        lst.append(n)
    n = n + 1
print(len(lst))
new = np.zeros(shape=(size, 3)) 
n = 0
for i in new:
 	i[1] = number  
 	i[2] = n
 	n = n+1
new_norm = np.block([new, normalized])

iota = os.path.join('/media/smiskey/USB/main_coil_0/Plots/{}'.format(inp), 'distances.txt')
iota_info = np.loadtxt(iota,dtype=float)
tmp = iota_info[0:size]
last = iota_info[-1]
new = np.zeros(shape=(size,4))
for i in range(len(new)):
    if i == 0:
        new[i]=last
    else:
        new[i]=tmp[i-1]
new_norm = np.block([new_norm,new])

### Aux Coil Current Database
os.chdir("/media/smiskey/USB/main_coil_0/{}".format(inp))
with h5py.File("metricdata.h5fd", 'w') as f:
 	f.create_dataset('dataset', data=new_norm)
     

os.chdir("/media/smiskey/USB/main_coil_0/{}".format(inp))
new = np.zeros(shape=(size, 3)) 
n = 0
for i in new:
 	i[1] = number
 	i[2] = n
 	n = n+1
input_data = os.path.join('/home/smiskey/Documents/HSX/HSX_Configs/auxCoil_Configs', 'auxStates{}.txt'.format("Predicted")) #Define auxState number
data = np.loadtxt(input_data, dtype = float)
new_data = np.block([new, data])
with h5py.File("coildata.h5fd", 'w') as f:
	f.create_dataset('dataset', data = new_data)



# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:54:35 2020

@author: micha
"""

import numpy as np
import h5py as hf

import os, sys
WORKDIR = os.path.join('/home', 'michael', 'Desktop', 'python_repos', 'turbulence-optimization', 'pythonTools')
sys.path.append(WORKDIR)

#import vmecTools.profile_analysis.profile_reader as pr

def advMetaNum(fileName):
    with open(fileName, 'r+') as file:
        for metaNum in file:
            pass
        file.write('\n{}'.format(int(metaNum)+1))

    return metaNum


def readStates(stateFile):
    states = []
    with open(stateFile, 'r') as file:
        for l, line in enumerate(file):
            line = line.strip()
            line = line.split()

            state = []
            for s in line:
                state.append(float(s))

            states.append(np.array(state))

    return np.array(states)


def readCrntConfig(conID, name='conData.txt'):
    """ Get the coil current configuration from the configuration ID.

    Parameters
    ----------
    conID : str
        configuration ID in form of {mainID}-{setID}-{jobID}.
    name : str, optional
        Name of the metafile where the main coil current data is stored. The
        default is 'conData.txt'.

    Returns
    -------
    main_crnt : arr
        Main coil current configuration, returned as normalized quantities.
    aux_crnt : arr
        Auxiliary coil current configuration, returned as normalized
        quantities.
    """
    conArr = conID.split('-')

    path = os.path.join('/mnt','HSX_Database','HSX_Configs','main_coil_'+conArr[0])
    filePath = os.path.join(path, name)

    main_crnt = np.empty(6)
    with open(filePath, 'r') as file:
        for l, line in enumerate(file):
            if l < 3:
                pass
            elif l==3:
                read = line[7:-2]
                read = read.split(', ')
                for r, rd in enumerate(read):
                    main_crnt[r] = float(rd)
            else:
                break

    pr_data = pr.profile_data(path, 'set_'+conArr[1], 'eps_effective')
    crnt_idx, data_idx = pr_data.find_config(conArr[2])
    aux_crnt = pr_data.crnt_profile[crnt_idx, pr_data.begIdx:]

    return main_crnt, aux_crnt


def findPathFromCrnt(crnt, base='D:\\HSX_Configs'):
    main_crnt = crnt[0:6]
    aux_crnt = crnt[6::]

    found, ID = checkMainConfig(base, main_crnt)

    if found:
        mainID = 'main_coil_{}'.format(ID)
        mainPath = os.path.join(base, mainID)

        set_path = [f.name for f in os.scandir(mainPath) if f.is_dir()]

        for setID in set_path:
            path_len = len([f.name for f in os.scandir( os.path.join(mainPath, setID) )])
            if path_len != 0:
                try:
                    prof_data = pr.profile_data(mainPath, setID, 'eps_effective')
                    idx = prof_data.find_crnt_profile(aux_crnt)
                    iD = int( prof_data.crnt_profile[idx,0] )
                    jobID = 'job_{}'.format(iD)

                    return os.path.join(base, mainID, setID, jobID)

                except ValueError:
                    print('Wout file not found in {0} inside {1}'.format(setID, mainID))
            else:
                raise IOError('Wout file for main coil current profile not found.')

        raise IOError('Wout file for main coil current profile not found.')

    else:
        raise IOError('Wout file for main coil current profile not found.')


def checkMainConfig(path, main):
    path_sub = [f.name for f in os.scandir(path) if f.is_dir()]
    for sub in path_sub:
        if sub[0:9] == 'main_coil':
            info_file = os.path.join(path, sub, 'conData.txt')

            main_chk = np.empty(6)
            with open(info_file, 'r') as file:
                for l, line in enumerate(file):
                    if l < 3:
                        pass
                    elif l==3:
                        read = line[7:-2]
                        read = read.split(', ')
                        main_chk = np.array([float(x) for x in read])
                    else:
                        break

            comp = np.sum( np.abs(main - main_chk) )
            if comp == 0:
                return True, sub[10::]
    return False, 0


def writeMetaFile(path, num, main, aux, base=10722.0, mult=14):
    nameTxt = os.path.join(path, 'conData.txt')
    nameH5 = os.path.join(path, 'conData.h5')

    job_num = str(aux.shape[0])
    set_id = 'set_{}'.format(num)

    aux_wID = np.empty((aux.shape[0], aux.shape[1]+1))
    for a, ax in enumerate(aux):
        aux_wID[a] = np.r_[a, ax]

    if os.path.exists(nameTxt):
        f = open(nameTxt, 'a')
        f.write(job_num+' Aux. Coil Configurations in '+set_id+'\n' +
                '(Current [A] = %.1f * %.1f * <percent>)\n\n' % (base, mult) )
        f.close()

        h5 = hf.File(nameH5, 'a')
        h5.create_dataset(set_id+'/config', data=aux_wID)
        h5.close()

    else:
        main_str = '[' + ', '.join(['{}'.format(i) for i in main]) + ']'

        f = open(nameTxt, 'w')
        f.write('Main Coil Multiplier\n' +
                '(Current [A] = %.1f * %.1f * <multiplier>)\n\n' % (base, mult) +
                '1-6 : ' + main_str + '\n\n' +
                str(job_num)+' Aux. Coil Configurations in '+set_id+'\n' +
                '(Current [A] = %.1f * %.1f * <percent>)\n\n' % (base, mult) )
        f.close()

        h5 = hf.File(nameH5, 'w')
        h5.create_dataset(set_id+'/config', data=aux_wID)
        h5.close()

if __name__ == '__main__':
    if True:
        ### Get Coil Currents From Configuration ID ###
        main_crnt, aux_crnt = readCrntConfig('60-1-0')

        #print(-10722.0 * main_crnt)
        #print(-10722.0 * 14.0 * aux_crnt)

        print(main_crnt)
        print(aux_crnt)

    if False:
        path = os.path.join('/mnt','HSX_Database','GENE','eps_valley','NL_TEM_jobs_20210721','NL_Select_conIDs.txt')

        conIDs = []
        main_crnts = []
        aux_crnts = []

        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines[1::]:
                conID = line.strip()
                main_crnt, aux_crnt = readCrntConfig(conID)

                conIDs.append(conID)
                main_crnts.append(main_crnt)
                aux_crnts.append(aux_crnt)

        new_path = os.path.join('/home','michael','Desktop','coil_currentst.txt')
        with open(new_path, 'w') as f:
            f.write("Main/Aux Coil Current in Amperes = -10722.0 * 14 * coilX\n")

            f.write("configID : main1 main2 main3 main4 main5 main6 : ")
            f.write("aux1 aux2 aux3 aux4 aux5 aux6 \n")
            for idx, conID in enumerate(conIDs):
                main_str = '\t'.join(['{0}'.format(ID) for ID in main_crnts[idx]])
                aux_str = '\t'.join(['{0}'.format(ID) for ID in aux_crnts[idx]])
                f.write(conID + " : " + main_str + " : " + aux_str + "\n")

    if False:
        ### Get Path From Current Configuration ###
        main_crnt = np.ones(6)
        aux_crnt = np.array([.1, .1, .1, .1, .1, .1])
        crnt = np.r_[main_crnt, aux_crnt]

        path = findPathFromCrnt(crnt)
        print(path)

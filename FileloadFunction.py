import os
import numpy as np
from preprocessing import EraseDuplicatedElect, GetHzStartEndIdxByElec, GetHzStartEndIdxByEMG, signal_mV, calc_y

def load(dir_path,person,part,ex_name,lv):

    if ex_name == 'Intensity':
        ex = 'Intensity/'
    else:
        ex = ex_name+'_'+str(lv)+'/'
        
    path = dir_path+'/'+ person+'/'+part+'/'+ex

    if ex_name == 'Intensity':
        key = [3,5,10,15,20,25,30]
    elif ex_name == 'Location':
        key = [1,2,3,4,5,6]
    elif ex_name == 'Angle':
        key = [0,90,180,270]
    dict_ = {}
    for  i,file in  enumerate(sorted([x for x in os.listdir(path) if x.endswith('.txt') and '이후' not in x])):
        tmp = file.split('.txt')[0].split('_')
        file_lines = [i.replace('\t', '-').split('-') for i in open(path+ file).readlines()]
        
        emg_raw = [int(i[2:]) for 
                            line in file_lines for i in line if i.strip().isdigit()]
        elect_raw = [int(i[:2] == '11') for 
                            line in file_lines for i in line if i.strip().isdigit()]
        emg_raw = np.array(emg_raw)
        elect_raw = np.array(elect_raw)

        elect_fixed = EraseDuplicatedElect(elect_raw)
        # start_idx, end_idx = GetHzStartEndIdxByElec(isElec=elect_fixed)
        idx = GetHzStartEndIdxByElec(isElec=elect_fixed)

        emg_raw = np.array(emg_raw[idx[0]-50:idx[0]+400])
        emg_raw = signal_mV(emg_raw,500)
        elect_fixed = np.array(elect_fixed[idx[0]-50:idx[0]+400])
        dict_[key[i]] = emg_raw

    return dict_
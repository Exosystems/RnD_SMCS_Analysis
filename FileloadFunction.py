import os
import numpy as np
import pandas as pd
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
    dict_emg = {}
    dict_elect = {}
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
        dict_emg[key[i]] = emg_raw
        dict_elect[key[i]] = elect_fixed

    return dict_emg, dict_elect

def load_whole_SMCS(dir, file):
    file_lines = [i.replace('\n', '-').split('-') for i in open(dir+ file).readlines()]
    emg_raw = [int(i)%1000 for 
                        line in file_lines for i in line if i.strip().isdigit()]
    elect_raw = [int(i)//1000 == 11 for 
                        line in file_lines for i in line if i.strip().isdigit()]
    emg_raw = np.array(emg_raw)
    elect_raw = np.array(elect_raw)

    elect_fixed = EraseDuplicatedElect(elect_raw)
    # start_idx, end_idx = GetHzStartEndIdxByElec(isElec=elect_fixed)
  
    # emg_raw = np.array(emg_raw[idx[0]-50:idx[0]+400])
    emg_raw = signal_mV(emg_raw,500)
    # elect_fixed = np.array(elect_fixed[idx[0]-50:idx[0]+400])
    return emg_raw, elect_fixed

def load_EMG(dir, file):

    file_lines = [i.replace('\n', '-').split(',') for i in open(dir+ file).readlines()]
    emg_raw = [int(i)for 
                        line in file_lines for i in line if i.strip().isdigit()]
    emg_raw = np.array(emg_raw)
    emg_raw = signal_mV(emg_raw,1100)
    return emg_raw

def load_listwdf(dir_path, people, parts):
    data_dict = {}
    data_dict_e = {}
    for person in people:
        Parts = {}
        Parts_e = {}
        for part in parts:
            Parts[part] = {}
            Parts_e[part] = {}
            
            Intensity = {}
            Intensity_ = {}
            Intensity[0], Intensity_[0] = load(dir_path,person,part,'Intensity',0)
                
            Angle = {}
            Angle_ = {}
            Angle[3], Angle_[3] = load(dir_path,person,part,'Angle',3)
            Angle[20], Angle_[20] = load(dir_path,person,part,'Angle',20)
            
            Location = {}
            Location_ = {}
            Location[3], Location_[3] = load(dir_path,person,part,'Location',3)
            Location[20], Location_[20] = load(dir_path,person,part,'Location',20)

            Parts[part]['Intensity'] = Intensity
            Parts[part]['Angle'] = Angle
            Parts[part]['Location'] = Location

            Parts_e[part]['Intensity'] = Intensity_
            Parts_e[part]['Angle'] = Angle_
            Parts_e[part]['Location'] = Location_

        data_dict[person] = Parts
        data_dict_e[person] = Parts_e


    data_list = []
    data_elist = []
    data_name_list = []
    data_name_df = []
    for person in people:
        for part in parts:
            for ex_name in data_dict[person][part].keys():
                for lv in data_dict[person][part][ex_name].keys():
                    for i in data_dict[person][part][ex_name][lv].keys():
                        # print(Whole_files[person][part][ex_name][lv][i].shape)
                        data_list.append(data_dict[person][part][ex_name][lv][i])
                        data_elist.append(data_dict_e[person][part][ex_name][lv][i])
                        data_name_list.append('_'.join([person,part, ex_name, str(lv),str(i)]))
                        data_name_df.append([person,part, ex_name, lv,i])
                        # print('_'.join([person,part, ex_name, str(lv),str(i)]))
    print(f'data_list shape: {np.shape(np.array(data_list))}')
    print(f'data_name_list shape: {np.shape(np.array(data_name_list))}')
    print(f'estimated data length: {6*3*(7+4*2+6*2)}')
    data_name_df = pd.DataFrame(data_name_df,columns=['person','part','ex_name','lv','i'])

    return data_dict, data_dict_e, data_list, data_elist, data_name_list, data_name_df

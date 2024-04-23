import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from preprocessing import EraseDuplicatedElect, GetHzStartEndIdxByElec, GetHzStartEndIdxByEMG, signal_mV, calc_y
from PlotFunction import Int_sub, Ang_sub, Loc_sub, SIMPLE
from FileloadFunction import load

'''
    Plot one signal
    num_sig = 1   # 1 for one signal, over 1 for several signals
'''

dir = './Result_experiments/240418_intloc_/CHLOE/biceps/Angle_20/'
file = 'Impulse_0_20240419183807_2_vastus medialis_right' 
col = []
num_sig = 1   # 1 for one signal, over 1 for several signals

# %matplotlib tk
emg_raw = SIMPLE(dir, file, 1 ,col)  

a = [0]+[3 if emg_raw[i]-emg_raw[i-1]>0.1 else 0 for i in range(1,len(emg_raw))]
start_idx = a.index(3)-1
upper_peaks, _ = find_peaks(emg_raw, height=2)
lower_peaks, _ = find_peaks([-x for x in emg_raw], height=0.5)

plt.plot(start_idx, emg_raw[start_idx], "x")
plt.plot(upper_peaks, emg_raw[upper_peaks], "x")
plt.plot(lower_peaks, emg_raw[lower_peaks], "x")


'''
    Plot whole people's intensity, angle, location experiment in one figure
'''

EX_name = '240418_intloc_'
EX_path  = './Result_experiments/' + EX_name + '/'

people = [x for x in os.listdir(EX_path) if x not in ['ETC','TEST', 'TMP','Fig']]
parts = ['quad', 'biceps', 'forearm']

# %matplotlib tk
for person in people:
    for part in parts:
        fig, axs = plt.subplot_mosaic([['a)', 'q)', 'q)', 'q)'], ['b)', 'w)', 'e)', 'r)'], ['c)','d)','f)', 'g)']],
                                    layout='constrained')


        Int_sub('./Result_experiments/240418_intloc_' ,person,part, ax = axs['a)'])
        Ang_sub('./Result_experiments/240418_intloc_' ,person,part,3, ax = axs['b)'])
        Ang_sub('./Result_experiments/240418_intloc_' ,person,part,20, ax = axs['e)'])
        Loc_sub('./Result_experiments/240418_intloc_' ,person,part,3, ax = [axs['c)'],axs['d)']],lim = [(0,500), (-3.5,3.5)])
        Loc_sub('./Result_experiments/240418_intloc_' ,person,part,3, ax = [axs['f)'],axs['g)']],lim = [(0,500), (-3.5,3.5)])

        fig.canvas.manager.set_window_title(person+'_'+part) 
plt.show()
plt.close()

'''
    Load whole signals in experiments as one dictionary data type
'''
dir_path = './Result_experiments/' + EX_name
Whole_files = {}
for person in people:
    Whole_files[person] = {}
    Parts = {}
    for part in parts:
        Parts[part] = {}
        
        Intensity = {}
        Intensity[0] = load(dir_path,person,part,'Intensity',0)
            
        Angle = {}
        Angle[3] = load(dir_path,person,part,'Angle',3)
        Angle[20] = load(dir_path,person,part,'Angle',20)
        
        Location = {}
        Location[3] = load(dir_path,person,part,'Location',3)
        Location[20] = load(dir_path,person,part,'Location',20)

        Parts[part]['Intensity'] = Intensity
        Parts[part]['Angle'] = Angle
        Parts[part]['Location'] = Location

    Whole_files[person] = Parts

'''
    making dictionary data into list, pandas data type
'''
data_list = []
data_name_list = []
data_name_pd = []
for person in people:
    for part in parts:
        for ex_name in Whole_files[person][part].keys():
            for lv in Whole_files[person][part][ex_name].keys():
                for i in Whole_files[person][part][ex_name][lv].keys():
                    data_list.append(Whole_files[person][part][ex_name][lv][i])
                    data_name_list.append('_'.join([person,part, ex_name, str(lv),str(i)]))
                    data_name_pd.append([person,part, ex_name, lv,i])
print(f'data_list shape: {np.shape(np.array(data_list))}')
print(f'data_name_list shape: {np.shape(np.array(data_name_list))}')
print(f'estimated data length: {6*3*(7+4*2+6*2)}')
data_name_pd = pd.DataFrame(data_name_pd,columns=['person','part','ex_name','lv','i'])
print(data_name_pd.head())
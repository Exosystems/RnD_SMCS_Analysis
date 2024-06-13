import numpy as np
from preprocessing import signal_mV, EraseDuplicatedElect, GetHzStartEndIdxByElec

'''
    This file contains an example of converting a value of SMCS to mv.
'''
def calc_y(x, amp):
    return (x-125) / 255 * ( 3.3 )  / amp  * 1000  # mV
    
def signal_mV(sig, amp):
    return np.array([calc_y(x,amp) for x in sig])


filename = ''
file_lines = [i.replace('\t', '-').split('-') for i in open(filename).readlines()]

emg_raw = [int(i[2:]) for 
                    line in file_lines for i in line if i.strip().isdigit()]
elect_raw = [int(i[:2] == '11') for 
                    line in file_lines for i in line if i.strip().isdigit()]
emg_raw = np.array(emg_raw)
elect_raw = np.array(elect_raw)

elect_fixed = EraseDuplicatedElect(elect_raw)
idx = GetHzStartEndIdxByElec(isElec=elect_fixed)

emg_raw = np.array(emg_raw)
emg_raw = signal_mV(emg_raw,500)
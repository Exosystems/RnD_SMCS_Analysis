import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from preprocessing import EraseDuplicatedElect, GetHzStartEndIdxByElec, GetHzStartEndIdxByEMG, signal_mV, calc_y
from PlotFunction import Int_sub, Ang_sub, Loc_sub




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
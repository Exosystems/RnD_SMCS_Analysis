import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import pandas as pd

from preprocessing import EraseDuplicatedElect, GetHzStartEndIdxByElec, GetHzStartEndIdxByEMG, signal_mV, calc_y
from PlotFunction import Int_sub, Ang_sub, Loc_sub, SIMPLE, plot_nxn
from FileloadFunction import load, load_listwdf

from scipy.signal import peak_prominences, savgol_filter
from scipy.interpolate import splrep, BSpline
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

EX_name = '240418_intloc_'
EX_path  = './Result_experiments/' + EX_name + '/'
dir_path = './Result_experiments/' + EX_name

people = [x for x in os.listdir(EX_path) if x not in ['ETC','TEST', 'TMP','Fig']]
parts = ['quad', 'biceps', 'forearm']

Whole_files, Whole_files_, data_list, data_elist, data_name_list, data_name_pd = load_listwdf(dir_path, people, parts)


df_ana = []
for idx in range(len(data_list)):
# for idx in range(30,50):
    dict_ = {}
    col = []
    num_sig = 1   # 1 for one signal, over 1 for several signals
    N = 1

    emg_raw, elect_fixed = SIMPLE([data_list[idx],'',data_name_list[idx]], 1 ,col)  
    start_mv = emg_raw[0]
    emg_raw = np.array([x-start_mv for x in emg_raw])
    filterd_emg = savgol_filter(emg_raw, window_length=5, polyorder=1)
    back = filterd_emg[200:]

    differences = np.diff(np.sign(filterd_emg))
    zero_crossings = np.where(differences)[0]

    threshold = 0.2
    gradient = np.gradient(emg_raw)
    large_changes_indices = np.where(np.abs(differences) > threshold)[0] + 1
    peaks = np.where(gradient > threshold)[0]
    peaks_minus = np.where(gradient < -threshold)[0]

    a = [0]+[3 if emg_raw[i]-emg_raw[i-1]>0.1 else 0 for i in range(1,len(emg_raw))]
    start_idx = a.index(3)-1
    upper_peaks, _ = find_peaks(filterd_emg, width=1, prominence=0.05)
    lower_peaks, _ = find_peaks([-x for x in filterd_emg],width=1, prominence=0.05)
    last_index = len(emg_raw) - list(emg_raw[::-1]).index(max(emg_raw)) - 1 # peaks_minus[0]

    plt.plot(filterd_emg, label = 'filtered')
    plt.plot(start_idx, emg_raw[start_idx], "x",color='r')
    plt.plot(upper_peaks, emg_raw[upper_peaks], "x",color='g')
    plt.plot(lower_peaks, emg_raw[lower_peaks], "x", color = 'b')
    plt.plot(last_index, emg_raw[last_index], "x",color = 'violet')
    dict_['start_real'] = np.where(data_elist[idx]  == 1)[0][0]
    dict_['start'] = start_idx+1
    dict_['end_e'] = last_index
    dict_['peaks'] = np.concatenate((upper_peaks,lower_peaks)) 

    if any(x > 200 for x in upper_peaks) or any(x > 200 for x in lower_peaks):

        derivatives = np.diff(back)
        derivatives = savgol_filter(derivatives, window_length=30, polyorder=1)
        threshold = 0.0051
        derivatives2 = np.diff(derivatives)
        derivatives2 = savgol_filter(derivatives2, window_length=30, polyorder=1)
        minima_indices = []
        for i in range(1, len(derivatives) - 1):
            if derivatives[i] >0 and derivatives[i +1] <0:
                minima_indices.append(i)

        upper_peaks, _ = find_peaks([-x for x in np.concatenate(([0 for i in range(200)],derivatives2))*1000], width=1, prominence=0.1)
        lower_peaks, _ = find_peaks([-x for x in np.concatenate(([0 for i in range(200)],derivatives))*100],width=1, prominence=0.3)
        plt.plot(upper_peaks, emg_raw[upper_peaks], "x",color='black')
        plt.plot(lower_peaks, emg_raw[lower_peaks], "x", color = 'black')

        dict_['after'] = np.concatenate((lower_peaks, upper_peaks))  
    # print(dict_)
    df_ana.append(dict_)
    plt.close()
plt.show()

print('done')


def clip(x):
    try:
        len(x)
        result = [i for i in x if i<300]
    except:
        result = x
    return result

def binary(x):
    try:
        len(x)
        result = 1
    except:
        result = 0
    return result
def func_mean(x):
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return int(np.mean(x))
    else:
        return 0
def interv(start, after):
    if after==0:
        return 0
    else:
        return after-start


df_ana_ = pd.DataFrame(df_ana)
df_ana_copy = df_ana_.copy()
df_ana_copy['after'] = df_ana_copy['after'].apply(lambda x: clip(x) )
df_ana_copy['A_after_ox'] = df_ana_copy['after'].apply(lambda x: binary(x) )     # after potential 있는지
df_ana_copy['front_peaks'] = df_ana_copy['peaks'].apply(lambda x: [i for i in x if i<200] )   # 앞에 피크 위치들
df_ana_copy['A_front_peaks_num'] = df_ana_copy['front_peaks'].apply(lambda x: len(x) )   # 앞에 피크 개수
df_ana_copy['after_peaks'] = df_ana_copy['peaks'].apply(lambda x: [i for i in x if i>220] )  # after potential peak 위치
df_ana_copy['A_after_peak'] = df_ana_copy['after_peaks'].apply(lambda x: func_mean(x) )  # after potential peak 위치


df_ana_copy['start'] = df_ana_copy.apply(lambda x: x['start']-x['start_real'], axis=1)
df_ana_copy['end_e'] = df_ana_copy.apply(lambda x: x['end_e']-x['start_real'], axis=1)
df_ana_copy['A_after_peak'] = df_ana_copy.apply(lambda x: interv(x['start_real'], x['A_after_peak']), axis=1)

df_ana_copy = df_ana_copy.drop(columns=['start_real','peaks','after','front_peaks','after_peaks'])


n_components = 2
n_clusters = 6
pca = PCA(n_components = n_components)
pca.fit(df_ana_copy)


data2 = pd.DataFrame(data = pca.transform(df_ana_copy), columns=['pc'+str(i) for i in range(1,n_components+1)])
kmeans = KMeans(n_clusters=6)
kmeans.fit(data2)
data2['labels'] = kmeans.predict(data2)

df = pd.concat([data_name_pd, df_ana_copy, data2],axis=1)


# for i in range(n_clusters):
#     plot_nxn(df[df['labels'] == i].index.values, df, Whole_files)

plot_dict = {}
plot_dict['intensity'] = [3,5,10,15,20,25,30]
plot_dict['angle'] = [0,90,180,270]
plot_dict['location'] = [1,2,3,4,5,6]

plot_dict['part'] = [ 'biceps', 'forearm', 'quad']

plot_dict['level'] = {}
plot_dict['level']['intensity'] = [0]
plot_dict['level']['angle'] = [3,20]
plot_dict['level']['location'] = [3,20]

plot_dict['string'] = {}
plot_dict['string']['intensity'] = ' Lv'
plot_dict['string']['angle'] = ' °'
plot_dict['string']['location'] = ' loc'

COL = 'start'

plot_list = ['intensity', 'angle', 'location']


for ex in plot_list:
    dict_ = {}
    for level in plot_dict['level'][ex]:
        for COL in ['start', 'end_e']:
            print(ex, level)
            plt.figure(ex+'_'+str(level)+'_'+COL)
            for i in plot_dict[ex]:
                c = []
                for p in plot_dict['part']:
                    c.append(df[(df['part']==p) & (df['lv']==level) & (df['i']==i)][COL].mean())
                dict_[p+'_'+str(i)] = c
                plt.plot(c, label = str(i)+plot_dict['string'][ex])
                plt.xticks([0,1,2],['biceps','forearm','quad'])
            plt.legend()
            if COL == 'end_e':
                plt.ylim(0,15)
            else:
                plt.ylim(0,3)
            plt.title(ex+'_'+str(level)+'_'+COL)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

from preprocessing import EraseDuplicatedElect, GetHzStartEndIdxByElec, GetHzStartEndIdxByEMG, signal_mV, calc_y

plt.rc('xtick', labelsize=7)  # x축 눈금 폰트 크기 
plt.rc('ytick', labelsize=7)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=7)  # 범례 폰트 크기

'''
    people: ['AA', 'BB', 'CC', 'DD', ... ,'ZZ']
    main_dir = './Result_experiments/date_exname_part/'

    for person in people:

        dir_path = main_dir + person +'/'
        save_path = main_dir+'Fig/'
        print(dir_path)
        print('INT')
        Int(dir_path,person,save_path, save=True)   # True, False -> save figure
        print('LOC')
        Loc(dir_path,person,save_path, save=True)
        print('ANG')
        Ang(dir_path,person,save_path, save=True)
        print('All')
        All(dir_path,'Location/5/',person, save_path, save=False)  
'''
extend_x = [(0,500),(45,110),(160,320)]
extend_y = [(-3.5,3.5),(-2,2),(-0.25,0.25)]

def Loc(dir_path,person,lv,save_path,lim,save):

    EMGdict = {}
    color = ['r','g','b']
    fig, ax = plt.subplots(1,2)
    num= 1
    pad = ['Proximal2Distal', 'Medial2Lateral']
    if lv == 0:
        location = 'Location/'
    else:
        location = 'Location_'+str(lv)+'/'
    path = dir_path+location

    for  i,file in  enumerate(sorted([x for x in os.listdir(path) if x.endswith('.txt')])):
        n = i//3
        
        tmp = file.split('.txt')[0].split('_')
        file_lines = [i.replace('\t', '-').split('-') for i in open(dir_path +location+ file).readlines()]
        
        emg_raw = [int(i[2:]) for 
                            line in file_lines for i in line if i.strip().isdigit()]
        elect_raw = [int(i[:2] == '11') for 
                            line in file_lines for i in line if i.strip().isdigit()]
        emg_raw = np.array(emg_raw)
        elect_raw = np.array(elect_raw)

        elect_fixed = EraseDuplicatedElect(elect_raw)
        # start_idx, end_idx = GetHzStartEndIdxByElec(isElec=elect_fixed)
        idx = GetHzStartEndIdxByElec(isElec=elect_fixed)

        emg_raw = np.array(emg_raw[idx[num]-50:idx[num]+400])
        emg_raw = signal_mV(emg_raw,500)
        elect_fixed = np.array(elect_fixed[idx[num]-50:idx[num]+400])

        emg_hz = [[] for _ in range(6)]
        RMS = [[] for _ in range(6)]
        is_print_data_len = True

        ax[n].plot(emg_raw, color=color[i%3], label=str(i%3+1))

        if len(tmp)>2:
            EMGdict[tmp[2]] = tmp[2]
        else:
            EMGdict[tmp[0]] = tmp[0]
            
        ax[n].title.set_text(pad[n])
        ax[n].set_xlim(lim[0][0],lim[0][1])
        ax[n].set_ylim(lim[1][0],lim[1][1])
        # ax[n].set_xlim(40,80)
        # ax[n].set_ylabel('(mV)')
        ax[n].legend()

    fig.canvas.manager.set_window_title(person+' M2D_'+str(lim[0][0])+'_'+str(lim[0][1])) 
    if save:
        plt.savefig(save_path+person+' M2D_'+str(lim[0][0])+'_'+str(lim[0][1])+'.png')
    return

def Loc_sub(dir_path,person,part,lv, ax, lim):

    EMGdict = {}
    color = ['r','g','b']
    num= 1
    pad = ['Proximal2Distal', 'Medial2Lateral']

    location = 'Location_'+str(lv)+'/'
    path = dir_path+'/'+ person+'/'+part+'/'+location

    for  i,file in  enumerate(sorted([x for x in os.listdir(path) if x.endswith('.txt') and '이후' not in x])):
        n = i//3
        
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

        emg_raw = np.array(emg_raw[idx[num]-50:idx[num]+400])
        emg_raw = signal_mV(emg_raw,500)
        elect_fixed = np.array(elect_fixed[idx[num]-50:idx[num]+400])

        emg_hz = [[] for _ in range(6)]
        RMS = [[] for _ in range(6)]
        is_print_data_len = True
        

        ax[n].plot(emg_raw, color=color[i%3], label=str(i%3+1))

        if len(tmp)>2:
            EMGdict[tmp[2]] = tmp[2]
        else:
            EMGdict[tmp[0]] = tmp[0]
            
        ax[n].title.set_text(pad[n])
        ax[n].set_xlim(lim[0][0],lim[0][1])
        ax[n].set_ylim(lim[1][0],lim[1][1])
        # ax[n].set_xlim(40,80)
        # ax[n].set_ylabel('(mV)')
        ax[n].legend()
    return

def Int(dir_path,person,save_path,save):
    
    location = 'Intensity/'
    
    path = dir_path+location
    EMGdict = {}
    color = ['r','g','b','k','#00A4E1','#00A52F','#FF7333']
    Hz = [3,5,10,15,20,25,30]
    fig, ax = plt.subplots(1,3)
    for num in range(3):

        for  i,file in  enumerate(sorted([x for x in os.listdir(path) if x.endswith('.txt')])):
            
            tmp = file.split('.txt')[0].split('_')
            file_lines = [i.replace('\t', '-').split('-') for i in open(dir_path +location+ file).readlines()]
            
            emg_raw = [int(i[2:]) for 
                                line in file_lines for i in line if i.strip().isdigit()]
            elect_raw = [int(i[:2] == '11') for 
                                line in file_lines for i in line if i.strip().isdigit()]
            emg_raw = np.array(emg_raw)
            elect_raw = np.array(elect_raw)

            elect_fixed = EraseDuplicatedElect(elect_raw)
            # start_idx, end_idx = GetHzStartEndIdxByElec(isElec=elect_fixed)
            idx = GetHzStartEndIdxByElec(isElec=elect_fixed)

            emg_raw = np.array(emg_raw[idx[num]-50:idx[num]+400])
            emg_raw = signal_mV(emg_raw,500)
            elect_fixed = np.array(elect_fixed[idx[num]-50:idx[num]+400])
            elect_fixed = [x for x in elect_fixed]

            emg_hz = [[] for _ in range(6)]
            RMS = [[] for _ in range(6)]
            is_print_data_len = True

            ax[num].plot(emg_raw,  label=str(Hz[i])+' LV', color=color[i])
            EMGdict[tmp[2]] = tmp[2]

        # ax[num].title.set_text('SMCS with Impulse')
        ax[num].legend()
        ax[num].set_ylim(extend_y[num][0],extend_y[num][1])
        ax[num].set_xlim(extend_x[num][0],extend_x[num][1])
        # ax[num].set_ylabel('(mV)')
    
    fig.canvas.manager.set_window_title(person+' Intensity') 
    if save:
        plt.savefig(save_path+person+' Intensity'+'.png')
    # plt.show()
    return

def Int_sub(dir_path,person,part, ax):
    
    location = 'Intensity/'
    
    path = dir_path+'/'+ person+'/'+part+'/'+location
    print(path)
    EMGdict = {}
    color = ['r','g','b','k','#00A4E1','#00A52F','#FF7333']
    Hz = [3,5,10,15,20,25,30]
    for  i,file in  enumerate(sorted([x for x in os.listdir(path) if x.endswith('.txt') and '이후' not in x])):
        
        tmp = file.split('.txt')[0].split('_')
        file_lines = [i.replace('\t', '-').split('-') for i in open(path+file).readlines()]
        
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
        elect_fixed = [x for x in elect_fixed]

        emg_hz = [[] for _ in range(6)]
        RMS = [[] for _ in range(6)]
        is_print_data_len = True

        ax.plot(emg_raw,  label=str(Hz[i])+' LV', color=color[i])
        EMGdict[tmp[2]] = tmp[2]

        ax.title.set_text('Intensity')
        ax.legend()
        ax.set_ylim(extend_y[0][0],extend_y[0][1])
        ax.set_xlim(extend_x[0][0],extend_x[0][1])
    
    return
        
def Ang(dir_path,person,save_path,save):
    
    location = 'Angle/'
    
    path = dir_path+location
    EMGdict = {}
    color = ['r','g','b','k']
    Hz = [0,90,180,270]
    fig, ax = plt.subplots(1,3)
    for num in range(3):

        for  i,file in  enumerate(sorted([x for x in os.listdir(path) if x.endswith('.txt')])):
            tmp = file.split('.txt')[0].split('_')
            file_lines = [i.replace('\t', '-').split('-') for i in open(dir_path +location+ file).readlines()]
            
            emg_raw = [int(i[2:]) for 
                                line in file_lines for i in line if i.strip().isdigit()]
            elect_raw = [int(i[:2] == '11') for 
                                line in file_lines for i in line if i.strip().isdigit()]
            emg_raw = np.array(emg_raw)
            elect_raw = np.array(elect_raw)

            elect_fixed = EraseDuplicatedElect(elect_raw)
            # start_idx, end_idx = GetHzStartEndIdxByElec(isElec=elect_fixed)
            idx = GetHzStartEndIdxByElec(isElec=elect_fixed)

            emg_raw = np.array(emg_raw[idx[num]-50:idx[num]+400])
            emg_raw = signal_mV(emg_raw,500)
            elect_fixed = np.array(elect_fixed[idx[num]-50:idx[num]+400])
            elect_fixed = [x for x in elect_fixed]

            emg_hz = [[] for _ in range(6)]
            RMS = [[] for _ in range(6)]
            is_print_data_len = True

            ax[num].plot(emg_raw,  label=str(Hz[i])+' °', color=color[i])
            EMGdict[tmp[2]] = tmp[2]

        # ax[num].title.set_text('SMCS with Impulse')
        ax[num].legend()
        ax[num].set_ylim(extend_y[num][0],extend_y[num][1])
        ax[num].set_xlim(extend_x[num][0],extend_x[num][1])
        # ax[num].set_ylabel('(mV)')
    fig.canvas.manager.set_window_title(person+' Angle') 
    if save:
        plt.savefig(save_path+person+' Angle'+'.png')
    # plt.show()
    return

def Ang_sub(dir_path,person,part,lv, ax):
    
    location = 'Angle_'+str(lv)+'/'
    
    path = dir_path+'/'+ person+'/'+part+'/'+location
    EMGdict = {}
    color = ['r','g','b','k']
    Hz = [0,90,180,270]
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
        elect_fixed = [x for x in elect_fixed]

        emg_hz = [[] for _ in range(6)]
        RMS = [[] for _ in range(6)]
        is_print_data_len = True

        ax.plot(emg_raw,  label=str(Hz[i])+' °', color=color[i])
        EMGdict[tmp[2]] = tmp[2]

        ax.title.set_text('Angle_'+str(lv))
        ax.legend()
        ax.set_ylim(extend_y[0][0],extend_y[0][1])
        ax.set_xlim(extend_x[0][0],extend_x[0][1])

    return

def All(dir_path,location, person,save_path,save, col):
    
    path = dir_path+location
    EMGdict = {}
    
    fig, ax = plt.subplots(1,3)
    for num in range(3):

        for  i,file in  enumerate(sorted([x for x in os.listdir(path) if x.endswith('.txt')])):
            
            tmp = file.split('.txt')[0].split('_')
            file_lines = [i.replace('\t', '-').split('-') for i in open(dir_path +location+ file).readlines()]
            
            emg_raw = [int(i[2:]) for 
                                line in file_lines for i in line if i.strip().isdigit()]
            elect_raw = [int(i[:2] == '11') for 
                                line in file_lines for i in line if i.strip().isdigit()]
            emg_raw = np.array(emg_raw)
            elect_raw = np.array(elect_raw)

            elect_fixed = EraseDuplicatedElect(elect_raw)
            # start_idx, end_idx = GetHzStartEndIdxByElec(isElec=elect_fixed)
            idx = GetHzStartEndIdxByElec(isElec=elect_fixed)

            emg_raw = np.array(emg_raw[idx[num]-50:idx[num]+400])
            emg_raw = signal_mV(emg_raw,500)
            elect_fixed = np.array(elect_fixed[idx[num]-50:idx[num]+400])
            elect_fixed = [x for x in elect_fixed]

            emg_hz = [[] for _ in range(6)]
            RMS = [[] for _ in range(6)]
            is_print_data_len = True
            if len(col)>0:
                ax[num].plot(emg_raw,  label= tmp[2]+tmp[-1], color = col[i])
            else:
                ax[num].plot(emg_raw,  label= tmp[2]+tmp[-1])
            EMGdict[tmp[2]] = tmp[2]

        ax[num].title.set_text('SMCS with Impulse')
        ax[num].legend()
        ax[num].set_ylim(-3.5,3.5)
        ax[num].set_xlim(0,500)
        # ax[num].set_ylabel('(mV)')
    
    fig.canvas.manager.set_window_title(person+' All') 
    if save:
        plt.savefig(save_path+person+' All'+'.png')
    # plt.show()
    return

def SIMPLE(data, num ,col):
    plt.figure()
    if type(data[0]) == type('str'):
        dir, file = data
        filename = dir+file+'.txt'

        tmp = file.split('.txt')[0].split('_')
        print(tmp[2])
        tmp = tmp[2]+'_'+tmp[-1]
        file_lines = [i.replace('\t', '-').split('-') for i in open(filename).readlines()]

        emg_raw = [int(i[2:]) for 
                            line in file_lines for i in line if i.strip().isdigit()]
        elect_raw = [int(i[:2] == '11') for 
                            line in file_lines for i in line if i.strip().isdigit()]
        emg_raw = np.array(emg_raw)
        elect_raw = np.array(elect_raw)

        elect_fixed = EraseDuplicatedElect(elect_raw)
        # start_idx, end_idx = GetHzStartEndIdxByElec(isElec=elect_fixed)
        idx = GetHzStartEndIdxByElec(isElec=elect_fixed)

        if num > 1:
            emg_raw = np.array(emg_raw)
            emg_raw = signal_mV(emg_raw,500)
            elect_fixed = np.array(elect_fixed)
        else:
            emg_raw = np.array(emg_raw[idx[0]-50:idx[0]+400])
            emg_raw = signal_mV(emg_raw,500)
            elect_fixed = np.array(elect_fixed[idx[0]-50:idx[0]+400])

        emg_hz = [[] for _ in range(6)]
        RMS = [[] for _ in range(6)]
        is_print_data_len = True
    else:
        emg_raw, elect_fixed, tmp = data
    if len(col)>0:
        plt.plot(emg_raw,  label= tmp, color = col[i])
    else:
        plt.plot(emg_raw,  label= tmp)
    plt.title('SMCS with Impulse')
    plt.legend()
    plt.ylim(-3.5,3.5)
    # plt.set_xlim(0,500)

    return emg_raw, elect_fixed

def plot_nxn(indexs, df, Whole_files):
    n = len(indexs)
    ro = int(np.ceil(np.sqrt(n )))
    co = int(np.ceil((n)/ro))
    fig, ax = plt.subplots(co,ro,figsize=(15,8))
    i = 0

    for i, key in enumerate(indexs):
        r,c =  i%co, i//co

        ax[r,c].title.set_text(df.iloc[key]['person']+'_'+
                df.iloc[key]['part'] +'_'+ 
                df.iloc[key]['ex_name']+ '_'+
                str(df.iloc[key]['lv'])+'_'+
                str(df.iloc[key]['i']))

        data = Whole_files[
            df.iloc[key]['person']][
                df.iloc[key]['part']][
                    df.iloc[key]['ex_name']][
                        int(df.iloc[key]['lv'])][
                            int(df.iloc[key]['i'])]
        
        ax[r,c].plot(data)
    plt.show()
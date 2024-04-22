import numpy as np
import copy 
import matplotlib.pyplot as plt
from typing import Tuple, List

def EraseDuplicatedElect(elect_raw: np.ndarray, window_size=5) -> np.ndarray:
    """
    5개 이상의 연속된 elect 는 모두 제거 (1에서 0으로 변경)
    개선된 신호, 지워진 부분, 제거된 elect 개수 return
    """
    elect_idx = []
    is_continuous = False
    num_continuous = 0
    num_erased = 0
    elect_fixed = copy.deepcopy(elect_raw)
    elect_erased = np.zeros_like(elect_raw)

    for idx in range(len(elect_fixed)):
        # 각 전기자극 시점에 대해
        if elect_fixed[idx] == 1:
            # 일단 해당 시점을 추가
            elect_idx.append(idx)
            # 예외처리
            if len(elect_idx) == 1:
                continue
            # 만약 전기자극이 연속적으로 들어왔다면 뒤의 전기 자극을 제거
            if elect_idx[-1] - elect_idx[-2] == num_continuous + 1:
                num_continuous += 1
                tmp = elect_idx.pop()
                elect_fixed[tmp] = 0
                elect_erased[tmp] = 1
                num_erased += 1
                if num_continuous > window_size:
                    is_continuous = True
                continue
            # 불연속적이더라도 5 sample 내로 전기자극이 또 들어왔다면 뒤의 전기 자극을 제거
            if elect_idx[-1] - elect_idx[-2] <= window_size and not is_continuous:
                tmp = elect_idx.pop()
                elect_fixed[tmp] = 0
                elect_erased[tmp] = 1
                num_erased += 1
        # 연속적인 전기자극이 끝난 경우 연속적인 전기자극의 시작 지점 제거
        elif elect_fixed[idx] == 0 and is_continuous:
            tmp = elect_idx.pop()
            elect_fixed[tmp] = 0
            elect_erased[tmp] = 1
            num_erased += 1
            num_continuous = 0
            is_continuous = False
        else:
            num_continuous = 0
            is_continuous = False

    return elect_fixed


def GetHzStartEndIdxByElec(isElec: np.ndarray, rest_threshold=850) -> Tuple[List[int], List[int]]:
    """전기 자극 구간 분리"""
    start_idx = []
    end_idx = []

    elect_idx = [i for i in range(len(isElec)) if isElec[i] == 1]
    # print(elect_idx)
    # print('se', start_idx,end_idx)
    try:
        start_idx.append(elect_idx[0])
    except:
        plt.clf()
        plt.plot(isElec)
        plt.show()
    for i, (m, n) in enumerate(zip(elect_idx, elect_idx[1:])):
        if elect_idx[i+1] > 61000:
            continue
        if n - m > rest_threshold:
            if 995 < n - m < 1005:
                continue
            start_idx.append(elect_idx[i + 1])
            end_idx.append(elect_idx[i])
    end_idx.append(elect_idx[-1])
    # return start_idx, end_idx
    return elect_idx


def GetHzStartEndIdxByEMG(emg: np.ndarray, p=75, rest_threshold=500) -> tuple:
    start_idx = []
    end_idx = []

    max_val = max(emg)
    elect_idx = [i for i in range(len(emg))
                 if emg[i] > max_val * p / 100
                 or emg[i] < max_val * (1 - p / 100)]
    start_idx.append(elect_idx[0])
    for i, (m, n) in enumerate(zip(elect_idx, elect_idx[1:])):
        if n - m > rest_threshold:
            if 995 < n - m < 1010:
                continue
            if m % 10000 < 5000:
                continue
            start_idx.append(elect_idx[i + 1])
            end_idx.append(elect_idx[i])
    end_idx.append(elect_idx[-1])
    return start_idx, end_idx

def calc_y(x, amp):
    return (x-125) / 255 * ( 3.3 )  / amp  * 1000  # mV
    
def signal_mV(sig, amp):
    return np.array([calc_y(x,amp) for x in sig])

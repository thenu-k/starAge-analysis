import numpy as np

def mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return mean, std

def get_threshold(A):
    diffs = np.abs(np.diff(A))
    # med = np.median(diffs)
    med = np.mean(diffs)
    _, std = mean_std(diffs)
    return med + 0 * std

def find_edge_debug(A, N_smooth=5, verbose=False):
    N = len(A)
    threshold = get_threshold(A)
    flag_count = 0
    i = N-5
    diffBuffer= [0]
    xBuffer = []
    yBuffer = []
    new = []
    while i > 0:   
        if len(xBuffer)>5:
            diffBuffer = diffBuffer[-5:]
            xBuffer = xBuffer[-5:]
            yBuffer = yBuffer[-5:]
        if verbose:
            print(f'---{i}')
            print(f'{A[i]}, {A[i-1]}')
        diff = abs(A[i] - A[i - 1])
        flag = (diff) >= (np.mean(diffBuffer) + np.std(diffBuffer)*5)
        if flag and xBuffer!=[]:
            # Noise extrapolation
            m, b = np.polyfit(xBuffer, yBuffer, deg=1)  
            min = 10
            pointRange = list(range(i-1, i-min, -1))
            x_new = np.array(pointRange)
            y_pred = m*x_new + b  
            y_actual = A[pointRange]
            bin = (y_actual > y_pred)*1
            #Mean
            bufferMean = np.mean(yBuffer)
            meanFlag = y_actual > bufferMean
            meanBin = meanFlag*1
            # Increasing
            y = A[pointRange]
            pointRange = np.array(pointRange)
            numVal = len(pointRange)
            newRange = pointRange-1
            yShift = A[newRange] - A[pointRange]
            yShiftBin = (yShift>0)*1
            # print(yShiftBin)
            if (np.sum(bin)==len(pointRange)) & (np.sum(meanBin)==len(pointRange)) & (np.sum(yShiftBin)>=numVal-3):
                return [i, [np.concatenate([xBuffer, x_new]), m*np.concatenate([xBuffer, x_new]) + b], True]
        diffBuffer.append(diff)
        xBuffer.append(i)
        yBuffer.append(A[i])
        i -= 1
    print("[FAIL] No edge found in input array.")
    return None, None, False

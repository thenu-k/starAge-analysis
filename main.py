from DUMP.GLOBAL import GLOBAL
GLOBAL = GLOBAL()
from astropy.io import fits
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from datetime import datetime
from DUMP.Analysis import Analysis
import pandas as pd
from itertools import combinations
from multiprocessing import Pool
import math

starData = pd.read_csv('allData.csv')
starData = starData.drop_duplicates(subset='SPECID')
flameMask = starData['flameMask']
interpolatedMask = starData['flameNewMask'] # Flame mask with interpolated ages
massOnlyInFlame = starData['flameMassMask']


analysis = Analysis(starData, GLOBAL)
universeAge = 13.8
analysis.errorQT(
    limitingAge=universeAge
)



# MASKS
ageMask = analysis.data['AGE'] > 10
alphaMask = analysis.alphaErrors > 5
problematicMask = ageMask & alphaMask
otherMask = ageMask & ~alphaMask

dataColumns = list(starData.columns)
parallaxMask = starData['PARALLAX'] > 0.2
alphaFeMask = starData['ALPHA_FE'] < 0.2

from DUMP.TESTING import allScatters, allHistograms, allLoss, allAltHistograms

######################
##### WRITE HERE #####
######################

# allScatters()
# allHistograms()
# allLoss()
# allHex()
# allAltHistograms()

from DUMP.TESTING import createCleanUpPlots, returnPI
from OTHER.mcmc_2 import MCMC_ANALYSIS

cleanUpMask = createCleanUpPlots(
    analysis=analysis,
    ageMask=ageMask,
    cleanUpValues=[3],
    flameMask=interpolatedMask
)

parallaxUnc = (starData['PARALLAX_ERROR']/ starData['PARALLAX']) <= 0.05
ipdMask = starData['ipd_frac_multi_peak'] <= 1
# mkMask = starData['MK_COMB'] < 2.5
ebvMask = starData['EBV'] > 0
finalMask = parallaxUnc & ipdMask & ebvMask & cleanUpMask

# exit()


# [samples, ll, [binCentres, PDF]] = MCMC_ANALYSIS(
#     mask=np.array([True]*len(starData)),
#     bins=100,
#     counts=[500, 2500],
#     writeTo='final',
# )
# pdf_test = samples["bin_mass"][0]
# [count, edgePoint] = analysis.edgeDetection(pdf_test)
# print(count, binCentres[counOUTPUT\\t])

reconVar = 'Test'
from pathlib import Path
Path(f"OUTPUT\\Recon-{reconVar}\\Edges").mkdir(parents=True, exist_ok=True)

# np.save(f'OUTPUT\\Recon-{reconVar}\\binMass.npy', samples["bin_mass"])
# np.save(f'OUTPUT\\Recon-{reconVar}\\ll.npy', ll)
# np.save(f'OUTPUT\\Recon-{reconVar}\\binCentres.npy', binCentres)
# np.save(f'OUTPUT\\Recon-{reconVar}\\media_pdf.npy', PDF)
# exit()

from OTHER.edgeDetection import find_edge_debug as edge
openVar = f'Recon-F'

ll = np.load(f'OUTPUT\\{openVar}\\ll.npy')
ys = np.load(f'OUTPUT\\{openVar}\\binMass.npy')
grid = np.load(f'OUTPUT\\{openVar}\\binCentres.npy')

index = np.argmax(ll)

t = ys[index]
ageList = []
llList = []
edgeList = []
print('-> Running Edge Detection')
for count in range(len(ys)):
    # break
    el = ys[count]
    llCurr = ll[count]
    [edgePoint, fit, found] =edge(el, 4)
    if not found: continue
    edgeList.append(edgePoint)
    grid = np.array(grid)
    prob = np.array(el)
    pointRange = list(range(len(grid)-1, edgePoint-10, -1))
    x = grid[pointRange]
    y = prob[pointRange]
    ageList.append(grid[edgePoint])
    llList.append(ll[count])
    plt.clf()
    plt.plot(x, y, marker='x', color='red', label='Probabilities')
    plt.plot(grid[fit[0]], fit[1], linestyle='--', color='black', label='Noise fit')
    plt.legend()
    plt.xlabel('Age')
    plt.ylabel('Probablity')
    plt.grid()
    plt.axvline(grid[edgePoint], linestyle='--', color='blue')
    plt.savefig(f'OUTPUT\\Recon-{reconVar}\\Edges\\{count}.png')
    plt.clf()
    plt.plot(grid, ys[count], marker='x', color='red')
    plt.grid()
    plt.xlabel('Age')
    plt.ylabel('Probability')
    plt.axvline(grid[edgePoint], linestyle='--', color='blue')
    plt.savefig(f'OUTPUT\\Recon-{reconVar}\\Edges\\{count}-full.png')


plt.close()
plt.clf()
heights, edges, _ = plt.hist(ageList, bins=50, histtype='step', color='blue')
indexMax = np.argmax(heights)
[a, b] = [edges[indexMax], edges[indexMax+1]]
tU = np.mean([a,b])
plt.axvline(tU, label=f'Mode= {tU}', linestyle='--', color='red')
plt.grid()
plt.legend()
plt.xlabel('Inferred t_U')
plt.ylabel('Frequency')
plt.savefig(f'OUTPUT\\Recon-{reconVar}\\tu.png')



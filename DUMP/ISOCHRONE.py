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
# import sqlite3
from multiprocessing import Pool
import math

# conn = sqlite3.connect('ALL_STARS.db')
starData = pd.read_csv('ALL_FLAME_ERR.csv')
starData = starData.drop_duplicates(subset='SPECID')
flameMask = starData['flameMask']
analysis = Analysis(starData, GLOBAL)
universeAge = 13.8
analysis.errorQT(
    limitingAge=universeAge
)
ageMask = analysis.data['AGE'] > 10
alphaMask = analysis.alphaErrors > 5
problematicMask = ageMask & alphaMask
otherMask = ageMask & ~alphaMask
starData['probMask'] = problematicMask

hdul = fits.open('ARCHIVE\\mist.fits')
mistData = hdul[1].data
columns = mistData.columns.names

logAges= mistData['LOGAGE_MIST']
mistAges = []
for logAge in logAges :
    mistAges.append(10**(logAge)/10**9)
mistAges = np.array(mistAges)
print(mistAges[0:10])
buffer = [mistData['SPECID'], mistAges]
mistFrame = pd.DataFrame(np.transpose(buffer), columns=['SPECID', 'AGE_MIST'])
print(mistFrame)
merged = pd.merge(starData, mistFrame, on='SPECID', how='inner')
print(merged)
ages_yy = merged['AGE']
ages_mist = merged['AGE_MIST']
plt.hexbin(ages_yy, ages_mist, gridsize=200)
plt.colorbar()
plt.xlim(0,20)
plt.ylim(0,20)
plt.xlabel('AGE YY')
plt.ylabel('AGE MIST')
plt.scatter(ages_yy[merged['probMask']], ages_mist[merged['probMask']], marker='x', color='red')
# print(len(ages_mist[merged['probMask']]))
plt.show()
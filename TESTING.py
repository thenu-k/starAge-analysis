from DUMP.GLOBAL import GLOBAL
GLOBAL = GLOBAL()
from astropy.io import fits
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from datetime import datetime
from Analysis import Analysis
import pandas as pd
from pathlib import Path

column_names = [
    "ID", "RA", "DEC", "AGE", "E_AGE", "FEH", "E_FEH", "FR",
    "JPHI", "JZ", "X", "Y", "Z", "SOURCE_ID", "SPEC_ID", "SN_G",
    "TEFF", "E_TEFF", "LOG_G", "E_LOG_G", "MK_SPEC", "E_MK_SPEC",
    "MK_COMB", "E_MK_COMB", "ALPHA_FE", "E_ALPHA_FE"
]

# data = pd.read_fwf('newStarSample.txt', names=column_names)
# # data.drop('SPEC_ID', axis=1)
# data.to_csv("cleanedDataPandas.csv", index=False)

from Analysis import Analysis



def createCleanUpPlots(analysis,ageMask, cleanUpValues, flameMask):
    # values = np.linspace(5, 1, 20)
    values = cleanUpValues
    numProbs = []
    numAll = []
    numFull = len(ageMask)
    for cleanUpValue in values:
        output = analysis.modelCorrelation(
            cleanUpValue=cleanUpValue,
            flameMask=flameMask
        )
        cleanUpMask = output['cleanUpMask']
        [slope, intercept] = output['interpolation']
        [numProb, numAlls] = output['num']
        [upperIntercept, lowerIntercept] = output['intercepts']
        numProbs.append(numProb)
        numAll.append(numAlls)
        [fig, ax1, ax2] = analysis.twoDimAnalysis(
            columns=['AGE', 'age_flame'],
            ALPHA_THRESHOLD=5,
            writeTo='OUTPUT\\MIST\\',
            AGE_CUT=0,
            xLim=[0,20],
            yLim=[0, 20],
            customProblematicMask=cleanUpMask,
            returnPlot=True
        )
        # plt.hexbin(analysis.data['age_flame'], analysis.data['AGE'], gridsize=30, cmap='Blues')
        xes = np.linspace(-5, 15, 100)
        def y(x):
            return slope*x + intercept  
        def yu(x):
            return slope*x*1 + upperIntercept
        def yl(x):
            return slope*x*1 + lowerIntercept
        ax1.plot(y(xes), xes)
        ax1.plot(yu(xes), xes, linestyle='--')
        ax1.plot(yl(xes), xes, linestyle='--')
        ax2.plot(yu(xes), xes, linestyle='--')
        ax2.plot(yl(xes), xes, linestyle='--')
        s = f"{cleanUpValue:.2f}".replace('.', ',')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.savefig(f'OUTPUT\\MIST\\value_new={s}_new.png')
    numAll = np.asarray(numAll)
    numProbs = np.asarray(numProbs)
    percs = numAll*100/numFull
    plt.clf()
    plt.grid()
    plt.scatter(values, numProbs, c=percs, cmap='plasma')
    plt.colorbar()
    plt.xlabel('Correlation factor')
    plt.ylabel('# problematic stars at end')
    plt.title('COLOR: final percentage of all stars')
    plt.savefig('OUTPUT\\MIST\\analysis.png')
    return cleanUpMask


#### Posterior Inferences
def returnPI(normalize=False):
    hdul = fits.open('ARCHIVE\\posteriorInferences.fits')
    rawData = hdul[1].data
    columns = list(rawData.columns.names)
    output = {
        'PIs': rawData['PDFAGE'],
        'ages': rawData['AGE'],
        'ageErrors': rawData['AGE_ERR'],
        'ageGrids': rawData['AGE_GRID'],
        'sourceIDs': rawData['SOURCE_ID'],
        'specIDs': rawData['SPECID'],
    }
    return output





def edgeDetection_test(PDF):
    PDF = np.array(PDF)
    pdfReversed = PDF[::-1]
    avgArray = [pdfReversed[0:5]]
    runningAvg = np.average(np.array(avgArray))
    count = 5
    edgePoint = pdfReversed[count]
    tripleFlag = 1
    while count <= len(pdfReversed):
        count += 1
        edgePoint = pdfReversed[count]
        if edgePoint > runningAvg:
            tripleFlag += 1
        if tripleFlag == 3:
            break
        avgArray.append(edgePoint)
        runningAvg = np.average(np.array(avgArray))
    return [count, edgePoint]


columns = columns = [
    "GaiaDR3",
    "RA_ICRS",
    "DE_ICRS",
    "Teff",
    "logg",
    "age",
    "Mini",
    "GMAG",
    "BPMAG",
    "RPMAG",
    "pflavour",
    "rgeo",
    "rpgeo"
]
# new = pd.read_csv('DUMP\\newComp.csv', delimiter=';')


def histogram(analysis, column):
    analysis.createHistogram(
        writeTo='OUTPUT\\TEST',
        columnName=column,
        outline=True,
        bins=10,
        save=True,
        xRange = 5,
        ALPHA_THRESHOLD=3
    )
def allHistograms(analysis, dataColumns):
    for columnName in dataColumns:
        try:
            if (columnName!='SPECID'):
                analysis.createHistogram(
                    writeTo='OUTPUT\\W5_HISTOGRAMS',
                    columnName=columnName,
                    outline=True,
                    bins=50,
                    save=True,
                    xRange = None,
                    ALPHA_THRESHOLD=5,
                    AGE_CUT=10
                )
        except Exception as e:
            print(f'-> Error in analysis: {e}')
def allScatters(analysis, dataColumns):
    correlationCoeffs = []
    heading = ['Parameter', 'Pearson','Absolute Value', 'P value']
    writeTo = 'OUTPUT\\W3_SCATTERS'
    for columnName in dataColumns:
        try:
            if (columnName!='SPECID'):
                cofs = analysis.oneDimAnalysis(
                    columnName=columnName, 
                    axLim=50, 
                    writeTo='OUTPUT\\W3_Scatters',
                    ALPHA_THRESHOLD = 5,
                    alphaAxLow=-50,
                    displayBeta=True,
                    xRange=10,
                    qualityCut=None,
                    AGE_CUT=10
                )
                correlationCoeffs.append([columnName, cofs[0], abs(cofs[0]) ,cofs[1]])
        except Exception as e:
            print(f'-> Error in analysis: {e}')
    with open(f'{writeTo}\\coeffs.csv', "w", newline='') as file:
                writer = csv.writer(file, delimiter="\t")
                writer.writerow(heading)
                writer.writerows(correlationCoeffs)
def scatter(analysis, column):
    cofs = analysis.oneDimAnalysis(
        columnName=column, 
        axLim=50, 
        writeTo='OUTPUT\\TEST',
        ALPHA_THRESHOLD = 3,
        alphaAxLow=-50,
        displayBeta=True,
        xRange=10,
        qualityCut=None
    )
def loss(analysis):
    analysis.lossCurve(
        columnName='MASS',
        ALPHA_THRESHOLD=5,
        writeTo='OUTPUT\\TEST'
    )
def allLoss(analysis, dataColumns):
    for columnName in dataColumns:
        try:
            if (columnName!='SPECID'):
                analysis.lossCurve(
                    columnName=columnName,
                    ALPHA_THRESHOLD=5,
                    writeTo='OUTPUT\\W5_ROC'
                )
        except Exception as e:
            print(f'-> Error in analysis: {e}')
def allAltHistograms(analysis=None, dataColumns=[], writeTo='TEST', customMask=None, Notes=None):
    for columnName in dataColumns:
        try:
            if (columnName!='SPECID'):
                analysis.alternateHistogram(
                    writeTo=f'OUTPUT\\{writeTo}',
                    columnName=columnName,
                    bins=50,
                    parameterRange=None,
                    ALPHA_THRESHOLD=5,
                    AGE_CUT=10,
                    customMask=customMask,
                    returnPlot=False,
                    normalize=False, 
                    Notes=Notes
                ) 
        except Exception as e:
            print(f'-> Error in analysis: {e}')
            
# def allHex(analysis, dataColumns):
#     # tempDF = starData.drop('SPECID', axis=1)
#     newCs = dataColumns.remove('SPECID')
#     newCs = dataColumns.remove('Unnamed: 0')
#     combins = list(combinations(dataColumns, 2))
#     endPoint = len(combins)
#     count2 = 263
#     for count in range(endPoint):
#         print(f'-> Progress: {count2}/{endPoint}')
#         try:
#             analysis.twoDimAnalysis(
#                 columns=[combins[count][0], combins[count][1]],
#                 ALPHA_THRESHOLD=5,
#                 writeTo='OUTPUT\\W3_HEX',
#                 AGE_CUT=10,
#                 silent=True
#             )
#         except Exception as e:
#             print(f'-> Error in analysis: {e}')  
#         count2 +=1
#     print('ALL DONE')
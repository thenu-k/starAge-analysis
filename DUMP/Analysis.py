from DUMP.GLOBAL import GLOBAL
GLOBAL = GLOBAL()
from astropy.io import fits
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
from datetime import datetime
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import spearmanr
import corner
import math
import pingouin as pg
from causallearn.utils.cit import CIT
# hdul = fits.open('Stellar_ages2.fits')
# data = hdul[1].data

class Analysis:
    def __init__(self, fitsData, GLOBAL):
        print('-> Initializing Analyser')
        self.GLOBAL = GLOBAL
        self.data = fitsData
        ages = self.data['AGE']
        self.length = len(ages)
        self.alphaErrors = None
        self.betaErrors = None
        self.comments = None
        print('-> Done')

    def tabulate(self, writeTo:str='tabulatedData.csv'):
        print('-> Tabulating Data')
        print('==')
        with open(writeTo, "w", newline='') as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(self.GLOBAL.COLUMNS)
            writer.writerows(self.data)

    def errorMetrics(self, age, uncertainty, limitingAge):
        alpha = (age-limitingAge)/uncertainty
        beta = (age)/uncertainty
        return [alpha, beta]
    
    def errorQT(self, data=None, limitingAge=15):
        print(f'-> Quantifying errors: limAge={limitingAge}')
        alphaErrors = []
        betaErrors = []
        ageData = np.array(self.data['AGE'])
        uncertaintyData = np.array(self.data['AGE_ERR'])
        np_alphaErrors = (ageData - limitingAge)/uncertaintyData
        np_betaErrors = (ageData)/uncertaintyData
        alphaErrors = np.nan_to_num(np_alphaErrors, nan=0.0, posinf=1e6, neginf=-1e6)
        betaErrors = np.nan_to_num(np_betaErrors, nan=0.0, posinf=1e6, neginf=-1e6)
        self.alphaErrors = alphaErrors
        self.betaErrors = betaErrors
        self.data['ALPHA_ERRORS'] = alphaErrors
        print('-> Done')
        return [alphaErrors, betaErrors]

    def oneDimAnalysis(self, columnName='AGE_ERR', limitingAge=13.8, axLim=50, writeTo='OUTPUT', recalculate=False, ALPHA_THRESHOLD = 5,  alphaAxLow = 0, plotFig=True, xRange=5, displayBeta = False, qualityCut=None, sortedData=None, AGE_CUT=10, customProblematicMask=None):
        print(f'-> Plotting errors vs. {columnName}')
        #OTHER DATA
        if sortedData is not None:
            data = np.array(sortedData)
        else:
            data = np.array(self.data[columnName])
        #ERROR CALCULATION
        os.makedirs(writeTo, exist_ok=True)
        if recalculate:
            [alphaErrors, betaErrors] = self.errorQT(limitingAge=limitingAge)
        else:
            alphaErrors = self.alphaErrors
            betaErrors = self.betaErrors
        alphaErrors = np.array(alphaErrors)
        betaErrors = np.array(betaErrors)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        #MASKING
        alphaMask = alphaErrors > ALPHA_THRESHOLD
        ageCutMask = self.data['AGE'] > AGE_CUT
        problematicMask = ageCutMask & alphaMask
        otherMask = ageCutMask & ~alphaMask 
        if customProblematicMask is not None:
            problematicMask = ageCutMask & customProblematicMask
            otherMask = ageCutMask & ~customProblematicMask 
        #QUALITY CUT
        if qualityCut!=None:
            qualityCutMask = (data > qualityCut) 
            totalStars = len(data[ageCutMask])
            remainingStars = len(data[qualityCutMask & ageCutMask])
            starsCut = len(data[~qualityCutMask & ageCutMask])
            if totalStars - remainingStars != starsCut:
                print('-> LOGIC ERROR WITH MASKS!!')
        #CORRELATION
        correlationCoeff = spearmanr(data[problematicMask], alphaErrors[problematicMask])  
        allMean = np.mean(data[problematicMask]) 
        allStd = np.std(data[problematicMask])
        [xMin, xMax] = [allMean-xRange*allStd, allMean+xRange*allStd]
        def plot():
            print('-> Drawing plots')  
            plt.clf()
            plt.scatter(data[otherMask], alphaErrors[otherMask], c=self.data['AGE'][otherMask], cmap='viridis', marker='x')
            plt.figtext(0.15, 0.85, f'Spearman:{round(correlationCoeff[0],3)} with p: {round(correlationCoeff[1], 3)}')
            if qualityCut!=None:
                plt.figtext(0.15, 0.80, f'Total Stars: {totalStars} ; After cut @ {qualityCut}: {remainingStars}')
            plt.colorbar()
            plt.scatter(data[problematicMask], alphaErrors[problematicMask], c=self.data['AGE'][problematicMask], cmap='plasma', marker='.')
            plt.colorbar()
            plt.xlabel(columnName)
            plt.ylabel(f'ALPHA; A_U={limitingAge}')
            plt.ylim(alphaAxLow, axLim)
            try:
                print('-> Normalising axes')
                # plt.xlim(xMin, xMax)
            except:
                print('-> ERROR: Cannot normalise axes')

            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
            plt.axhline(y=ALPHA_THRESHOLD, color='red', linewidth=1, linestyle='-')
            if qualityCut!=None:
                plt.axvline(x=qualityCut, color='black', linewidth=1, linestyle='dashed')
            plt.savefig(os.path.join(writeTo, f'alphaErrors_PARAMETER=_{columnName}_UNIVERSEAGE=_{limitingAge}_AXLIMIT={axLim}_ALPHATHRESHOLD=_{ALPHA_THRESHOLD}_TIME={timestamp}.png'))
            if displayBeta:
                plt.clf()
                plt.scatter(data[otherMask], betaErrors[otherMask], c=self.data['AGE'][otherMask], cmap='viridis', marker='x')
                plt.colorbar()
                plt.scatter(data[problematicMask], betaErrors[problematicMask], c=self.data['AGE'][problematicMask], cmap='plasma', marker='.')
                plt.colorbar()
                plt.ylim(0, axLim)
                # plt.xlim(xMin, xMax)
                plt.xlabel(columnName)
                plt.ylabel(f'BETA')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
                plt.savefig(os.path.join(writeTo, f'betaErrors_PARAMETER=_{columnName}_UNIVERSEAGE=_{limitingAge}_AXLIMIT={axLim}_ALPHATHRESHOLD=_{ALPHA_THRESHOLD}_TIME={timestamp}.png'))
                plt.clf()
            return correlationCoeff
        if plotFig:
            plot()
        print('-> Done')
        return correlationCoeff

    def createHistogram(self, columnName='AGE', writeTo='HISTOGRAM', bins=40, outline=True, proportions=True, save=True, xRange= None, ALPHA_THRESHOLD=5, AGE_CUT=10, parameterRange = [0,100]):
        print(f'-> Plotting histogram of {columnName}')
        os.makedirs(writeTo, exist_ok=True)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        alphaErrors = np.array(self.alphaErrors)
        #MASKING
        alphaMask = alphaErrors > ALPHA_THRESHOLD
        ageCutMask = self.data['AGE'] > AGE_CUT
        parameterMask = (self.data[columnName] >= parameterRange[0]) & (self.data[columnName] <= parameterRange[1])
        problematicMask = ageCutMask & alphaMask & parameterMask
        otherMask = ageCutMask & ~alphaMask & parameterMask
        groupedColumns = [self.data[columnName][problematicMask], self.data[columnName][otherMask]]
        labels = [f'Alpha > {ALPHA_THRESHOLD}', f'Alpha < {ALPHA_THRESHOLD}']
        colors = [
                # '#4E79A7', 
                # '#F28E2B',  
                '#E15759',
                # '#76B7B2',  
                '#59A14F'
        ]  
        plt.clf()
        fig, ax = plt.subplots(1, figsize=(16, 8))
        if xRange !=None:
            try:
                allMean = np.mean(self.data[columnName][otherMask]) 
                allStd = np.std(self.data[columnName][otherMask])
                [xMin, xMax] = [allMean-xRange*allStd, allMean+xRange*allStd]
                ax.set_xlim(left=10, right=20)
            except:
                print('-> ERROR: Cannot set x limits')
        ax2 = ax.twinx()
        #BINS
        all_data = self.data[columnName][otherMask | problematicMask]
        # bin_edges = np.histogram_bin_edges(all_data, bins=40)  # or use np.linspace 
        if outline:
            ax.hist(self.data[columnName][otherMask], edgecolor = "black", color=colors[1], label=f'Alpha < {ALPHA_THRESHOLD}',  alpha=0.5, bins=bins)
            ax2.hist(self.data[columnName][problematicMask], edgecolor = "black", color=colors[0], label=f'Alpha > {ALPHA_THRESHOLD}',  alpha=0.5, bins=bins)
        else:
            ax.hist(self.data[columnName][otherMask], bins=bins, histtype='bar', color=colors[1], label=f'Alpha < {ALPHA_THRESHOLD}',  alpha=0.5)
            ax2.hist(self.data[columnName][problematicMask], bins=bins, histtype='bar', color=colors[0], label=f'Alpha > {ALPHA_THRESHOLD}',  alpha=0.5)
        ax.set_xlabel(columnName)
        ax2.set_ylabel(f'FREQ: ALPHA > {ALPHA_THRESHOLD}', color=colors[0])
        ax.set_ylabel(f'FREQ: ALPHA < {ALPHA_THRESHOLD}', color=colors[1])
        ax2.tick_params('y', colors=colors[0])
        ax.tick_params('y', colors=colors[1])
        plt.grid(True, alpha=0.3)

        if save:
            plt.savefig(os.path.join(writeTo, f'histogram_PARAMETER=_{columnName}_TIME={timestamp}.png'))
        print('-> Done')

    def lossCurve(self, columnName='AGE', ALPHA_THRESHOLD=5, writeTo='OUTPUT'):
        print(f'-> Creating LOSS curve for: {columnName}')
        os.makedirs(writeTo, exist_ok=True)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        data = np.array(self.data[columnName])
        alphaErrors = np.array(self.alphaErrors)
        alphaMask = alphaErrors > ALPHA_THRESHOLD
        ageCutMask = self.data['AGE'] > 10
        xMax = max(data)
        xMin = min(data)
        parameterSpace = np.linspace(xMin, xMax, num=100)
        numTotal = len(data[ageCutMask])
        currentFractions_Forward = []
        problematicFractions_Forward = []
        currentFractions_Backward = []
        problematicFractions_Backward = []
        for cutPoint in parameterSpace:
            cutMask_Forward = data > cutPoint
            cutMask_Backward = data < cutPoint
            currentStars_Forward = data[ageCutMask & cutMask_Forward]
            currentStars_Backward = data[ageCutMask & cutMask_Backward]
            problematicStars_Forward = data[ageCutMask & cutMask_Forward & alphaMask]
            problematicStars_Backward = data[ageCutMask & cutMask_Backward & alphaMask]
            numCurr_Forward = len(currentStars_Forward)
            numProblem_Forward = len(problematicStars_Forward)
            numCurr_Backward = len(currentStars_Backward)
            numProblem_Backward = len(problematicStars_Backward)  
            try:
                problematicFractions_Forward.append(numProblem_Forward*100/numCurr_Forward)
                currentFractions_Forward.append(numCurr_Forward*100/numTotal)
            except:
                currentFractions_Forward.append(numCurr_Forward*100/numTotal)
                problematicFractions_Forward.append(0)
            try:
                problematicFractions_Backward.append(numProblem_Backward*100/numCurr_Backward)
                currentFractions_Backward.append(numCurr_Backward*100/numTotal)           
            except:
                currentFractions_Backward.append(numCurr_Backward*100/numTotal)
                problematicFractions_Backward.append(0)
        plt.clf()
        plt.scatter(currentFractions_Forward, problematicFractions_Forward, c=parameterSpace, cmap='plasma')
        plt.colorbar()
        plt.title(f'ROC FOWARD: {columnName}')
        plt.xlabel('% of current dataset')
        plt.ylabel('% of problematic stars')
        plt.savefig(os.path.join(writeTo, f'LOSS_PARAMETER=_{columnName}_FORWARD_TIME={timestamp}.png'))
        plt.clf()
        plt.scatter(currentFractions_Backward, problematicFractions_Backward, c=parameterSpace, cmap='plasma')
        plt.colorbar()
        plt.title(f'ROC BACKWARD: {columnName}')
        plt.xlabel('% of current dataset')
        plt.ylabel('% of problematic stars')
        plt.savefig(os.path.join(writeTo, f'LOSS_PARAMETER=_{columnName}_BACKWARD_TIME={timestamp}.png'))       
        print('-> Done')
    
    def twoDimAnalysis(self, columns=['AGE, MASS'], ALPHA_THRESHOLD=5, writeTo='OUTPUT', AGE_CUT=10, silent=True, yLim=None, xLim=None, customMask=None, customProblematicMask=None, returnPlot=False):
        if silent==False:
            print(f'-> Creating Hex bins for: {columns[0]} and {columns[1]}')
        os.makedirs(writeTo, exist_ok=True)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        xAll = self.data[columns[0]]
        yAll = self.data[columns[1]]
        alphaErrors = self.alphaErrors
        alphaMask = alphaErrors > ALPHA_THRESHOLD
        ageCutMask = self.data['AGE'] > AGE_CUT
        problematicMask = ageCutMask & alphaMask
        otherMask = ageCutMask & ~alphaMask 
        if customMask is not None:
            problematicMask = ageCutMask & alphaMask & customMask
            otherMask = ageCutMask & ~alphaMask & customMask
        if customProblematicMask is not None:
            # problematicMask =  ~customProblematicMask
            otherMask =  [True]*len(alphaErrors) 
            otherMask = customProblematicMask
        x_allMean = np.mean(xAll[otherMask]) 
        x_allStd = np.std(xAll[otherMask])
        z = 5
        if xLim is not None:
            [xMin, xMax] = xLim
        else:
            [xMin, xMax] = [x_allMean-z*x_allStd, x_allMean+z*x_allStd]  
        y_allMean = np.mean(yAll[otherMask]) 
        y_allStd = np.std(yAll[otherMask])
        if yLim is not None:
            [yMin, yMax] = yLim
        else:
            [yMin, yMax] = [y_allMean-z*y_allStd, y_allMean+z*y_allStd]  
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        xRangeMask = (xAll >= xMin) & (xAll <=xMax)
        yRangeMask = (yAll >= yMin) & (yAll <=yMax)
        hb1 = ax1.hexbin(xAll[otherMask & xRangeMask & yRangeMask], yAll[otherMask & yRangeMask & xRangeMask], gridsize=30, cmap='Blues')
        ax1.set_xlabel(columns[0])
        ax1.set_ylabel(columns[1])
        ax1.set_xlim(xMin, xMax)
        ax1.set_ylim(yMin, yMax)
        ax1.set_title(f'ALL: {columns[0]}_&&_{columns[1]}')
        fig.colorbar(hb1, ax=ax1)
        hb2 = ax2.hexbin(xAll[problematicMask & xRangeMask & yRangeMask], yAll[problematicMask & xRangeMask & yRangeMask], gridsize=30, cmap='Greens')
        fig.colorbar(hb2, ax=ax2)
        ax2.set_xlim(xMin, xMax)
        ax2.set_ylim(yMin, yMax)
        ax2.set_title(f'PROBLEMATIC ONLY')
        plt.tight_layout()
        plt.text(2, 5, f'All stars: {len(xAll[otherMask])}; Problematic: {len(xAll[problematicMask])}', fontsize=12, color='red')
        if returnPlot:
            if silent==False:
                print('-> Done')
            return [fig, ax1, ax2]
        else:
            plt.savefig(os.path.join(writeTo, f'HEX_PARAMETERS=_{columns[0]}_&&_{columns[1]}_TIME={timestamp}.png')) 
            if silent==False:
                print('-> Done')

    def alternateHistogram(self, columnName='AGE', writeTo='HISTOGRAM', bins=40, ALPHA_THRESHOLD=5, AGE_CUT=10, parameterRange = None, customMask=None, returnPlot=False, customData=None, Note=None, normalize=False, Notes=None):
        print(f'-> Plotting histogram of {columnName}')
        os.makedirs(writeTo, exist_ok=True)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        alphaErrors = np.array(self.alphaErrors)
        #MASKING
        if customData is not None:
            data = customData
        else:
            data = self.data[columnName]
        if parameterRange is None:
            z=5
            x_allMean = np.mean(data) 
            x_allStd = np.std(data)
            [xMin, xMax] = [x_allMean-z*x_allStd, x_allMean+z*x_allStd]
            parameterRange = [xMin, xMax]
        if customMask is not None:
            print('-> Using Custom Mask')
        else:
            print('-> Using default masks')
            alphaMask = alphaErrors > ALPHA_THRESHOLD
            ageCutMask = self.data['AGE'] > AGE_CUT
            parameterMask = (data >= parameterRange[0]) & (data <= parameterRange[1])
            problematicMask = ageCutMask & alphaMask & parameterMask
            otherMask = ageCutMask & ~alphaMask & parameterMask 
        parameterSpace = np.linspace(parameterRange[0], parameterRange[1], bins)
        binWidth = parameterSpace[-1]-parameterSpace[0]/len(parameterSpace)
        dx = parameterSpace[1] - parameterSpace[0]
        binCentres = []
        if customMask is None:
            allHeights = []
            problematicHeights = []
            if normalize:
                normalizingHeights = []
        else:
            heights = [[] for _ in range(len(customMask))] 
        for point in parameterSpace:
            beforePoint = point
            afterPoint = point + dx
            binCentre = (afterPoint + beforePoint)/2
            binCentres.append(binCentre)
            currentMaskAll = (data >= beforePoint) & (data <= afterPoint) 
            if customMask is None:
                allHeights.append(len(data[currentMaskAll & otherMask]))
                problematicHeights.append(len(data[currentMaskAll & problematicMask]))     
                if normalize:
                    normalizingHeights.append(len(data[currentMaskAll]))
            else:
                maskCount = 0 
                for mask in customMask:
                    currHeight = len(data[currentMaskAll & mask])
                    heights[maskCount].append(currHeight)
                    maskCount +=1 
        if normalize:
            allHeights = np.divide(allHeights, normalizingHeights)
            problematicHeights  = np.divide(problematicHeights, normalizingHeights)
        print(f'-> Done')
        if not returnPlot:
            if customMask is None:
                plt.clf()
                fig, ax = plt.subplots()
                ax2 = ax.twinx()
                ax2.plot(binCentres, problematicHeights, label='Problematic freq', color='red', marker='o', zorder=2)
                ax.plot(binCentres, allHeights, label='All freq', color='green', marker='x', zorder=3)
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1.15, 1))
                ax.grid(visible=True, alpha=0.5, axis='both')
                ax.set_title(f'Histogram of {columnName}')   
                ax.set_ylabel('# Stars', color='green')
                ax2.set_ylabel('# Stars', color='red')
                plt.savefig(os.path.join(writeTo, f'alternate_histogram_PARAMETER=_{columnName}_TIME={timestamp}.png'))
                return [fig, ax, ax2]
            else:
                fig, ax = plt.subplots()
                ax.grid(alpha=0.5, axis='both') 
                fig.set_size_inches(14, 8)
                ax.set_title(f'Histogram of {columnName}')
                heights = np.array(heights)
                count = 0
                for heightSet in heights:
                    normCoeff = np.sum(heightSet) * binWidth
                    sci = f"{normCoeff:.2e}"
                    heightsNormed = heightSet / normCoeff 
                    ax.plot(binCentres, heightsNormed, marker='.', label=f'{Notes[count]} ; Actual Area = {sci}')
                    count += 1
                ax.set_xlabel(columnName)
                ax.set_ylabel('Normalised Probability')
                ax.legend()
                # plt.show() 
                plt.savefig(os.path.join(writeTo, f'alternate_histogram_PARAMETER=_{columnName}_TIME={timestamp}.png'), dpi=300) 
                return [fig, ax, heights]
        else:
            if customMask is None:
                return [binCentres, problematicHeights, allHeights]
            else:
                return [binCentres, heights]
    
    def correlations(self, ALPHA_THRESHOLD=5, AGE_CUT=5, parameterSet=['AGE', 'MASS', 'ALPHA'], printOn=False):
        if printOn:
            print(f'-> Calculating correlations of: {parameterSet[0]} & {parameterSet[1]} | {parameterSet[2]}')
        x = self.data[parameterSet[0]]
        y = self.data[parameterSet[1]]
        z = self.data[parameterSet[2]]
        alphaMask = self.alphaErrors > ALPHA_THRESHOLD
        ageCutMask = self.data['AGE'] > AGE_CUT
        problematicMask = ageCutMask & alphaMask 
        otherMask = ageCutMask & ~alphaMask    
        [x_all, y_all, z_all] = [x[otherMask], y[otherMask], z[otherMask]] 
        [x_prob, y_prob, z_prob] = [x[problematicMask], y[problematicMask], z[problematicMask]] 
        allArray = np.asarray([x_all, y_all, z_all])
        problematicArray = np.asarray([x_prob, y_prob, z_prob])
        kci_all = CIT(allArray, method="kci")
        kci_prob = CIT(problematicArray, method='kci')
        p_all = kci_all(0, 1, [2])
        p_prob = kci_prob(0, 1, [2])
        if printOn:
            print(f'-> Done')
        return [p_all, p_prob]

    def modelCorrelation(self, cleanUpValue=3., ALPHA_THRESHOLD=5, flameMask=None):
        print('-> Cleaning up models via correlation')
        alphaMask = self.alphaErrors > ALPHA_THRESHOLD
        ages = self.data['AGE']
        flameAges = self.data['age_flame']
        maskHistory = [[True]*len(self.data['AGE'])]
        if flameMask is not None:
            maskHistory = [flameMask]
        prevZsum = 0
        convergenceLimit = 0.05
        iteration = 0
        zDiff = 100
        ageMask = ages > 10
        print(f'-> Number of intial all stars: {len(ages[maskHistory[-1]])}')
        print(f'-> Number of intial older stars: {len(ages[maskHistory[-1] & ageMask])}')
        print(f'-> Number of intial problematic stars: {len(ages[maskHistory[-1] & ageMask & alphaMask])}')
        errorOut = None
        while zDiff > convergenceLimit and iteration < 100:
            maskIndices = []
            try:
                slope, intercept = np.polyfit(flameAges[maskHistory[-1]], ages[maskHistory[-1]], 1)
            except:
                print('Failed')
                print(iteration) 
                print(flameAges[maskHistory[-1]][:10])
                print(ages[maskHistory[-1]][:10])
                exit()
            errors = np.abs(slope*flameAges[maskHistory[-1]] - ages[maskHistory[-1]] + intercept)/np.sqrt(slope**2 + 1)
            errStd = np.std(errors)
            errM = np.mean(errors)
            mask = (((errors-errM)/errStd) < cleanUpValue) 
            errorOut = errors.copy()[mask]
            zscores = (errors-errM)/errStd
            zsum = np.sum(np.abs(zscores))
            zDiff = abs((zsum-prevZsum)*100/prevZsum)
            prevZsum = zsum 
            maskIndices = np.where(maskHistory[-1])[0]
            fullLengthMask = np.zeros_like(maskHistory[-1], dtype=bool)
            fullLengthMask[maskIndices] = mask
            maskHistory.append(fullLengthMask & maskHistory[-1])
            iteration += 1
            upperIntercept = intercept + 3*errStd*np.sqrt(1+slope**2)
            lowerIntercept = intercept - 3*errStd*np.sqrt(1+slope**2)
        print(f'-> Number of final all stars: {len(ages[maskHistory[-1]])}')
        print(f'-> Number of final problematic stars: {len(ages[maskHistory[-1] & alphaMask])}')
        print(f'-> Number of final older stars: {len(ages[maskHistory[-1] & ageMask])}')
        print(f'-> Iterations: {iteration}')
        print('-> DONE')
        output = {
            "cleanUpMask": maskHistory[-1],
            "interpolation": [slope, intercept],
            "num": [len(ages[maskHistory[-1] & alphaMask]),len(ages[maskHistory[-1]])],
            "intercepts": [upperIntercept, lowerIntercept],
            "errors": errorOut
        }
        return output
    
    def edgeDetection(self, PDF):
        PDF = np.array(PDF)
        pdfReversed = PDF[::-1]
        avgArray = pdfReversed[0:5]
        runningAvg = np.average(avgArray)
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
            avgArray = np.append(avgArray, edgePoint)
            runningAvg = np.average(avgArray)
        return [len(PDF)-count, edgePoint]

    def binningFLAME(self, bins):
        print('-> Binning in FLAME')
        flameMask = self.data['flameMask']
        yyAges = self.data['AGE'][flameMask]
        flameAges = self.data['age_flame'][flameMask]
        yyBins = np.linspace(np.min(yyAges), np.max(yyAges), 50)
        flameBins = np.linspace(np.min(flameAges), np.max(flameAges), 50)
        meanYY = []
        currFlame = []
        pYY = []
        mYY = []
        stds = []
        starCount = []
        prev = 0
        count = 0
        for el in flameBins:
            if el==0: continue
            currMask = (flameAges>=prev) & (flameAges<el)
            relevantYY = yyAges[currMask]
            starCount.append(len(relevantYY))
            mean = np.mean(relevantYY)
            std = np.std(relevantYY)
            meanYY.append(mean)
            currFlame.append(np.mean([prev, el]))
            pYY.append(mean + 3*std)
            mYY.append(mean - 3*std)
            stds.append(std)
            prev = el
            count +=1
        output = {
            'meanYY': meanYY,
            'currFLAME': currFlame,
            'std': std,
            'pYY': pYY,
            'mYY': mYY,
            'starCount': starCount
        }
        print('-> Done')
        return output



            












        
        


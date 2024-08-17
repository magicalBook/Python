#!/usr/bin/env python
# coding: utf-8
                
import numpy as np
import importlib

def importPackage(packageName, onlyMessage=False):
    import importlib
    packageSpec = importlib.util.find_spec(packageName)
    if packageSpec == None:
        if onlyMessage:
            print('The package, {packageName}, is not installed. Install {packageName} first..')
            raise StopException
        else:            
            print('The package, {packageName}, is not installed. Installing {packageName}..')
            import sys
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", packageName])

importPackage('scipy')

from scipy.interpolate import CubicSpline,PchipInterpolator,CubicHermiteSpline,Akima1DInterpolator,UnivariateSpline,krogh_interpolate,BarycentricInterpolator
from scipy.interpolate import splev, splrep, interp1d
from scipy.signal import savgol_filter
from numpy.random import default_rng

try: rng
except NameError:
    rng = default_rng(54321)

def genDataRead(dataPath=''):
    print('reading data from files..') 
    dataPeak=np.load(dataPath+'Peak.npy',allow_pickle=True)
    baseData=np.load(dataPath+'Base.npy',allow_pickle=True)
    print('reading done..') 
    return dataPeak, baseData       

# 0.025 <= peakMinHeight< 0.15
def genData(dataPath='', nData=500000, lenSignal=512, baseOrder=7, peakCount=(8, 24), peakMinHeight=0.025, peakWidth=(5,25)):

    nOrder = baseOrder
    minPeak, maxPeak = peakCount
    
    minPeakWidth, maxPeakWidth = peakWidth
    widthNum = 4 # 4 구간
    widthPos = np.around( np.linspace(minPeakWidth, maxPeakWidth, num = widthNum+1 ) ) 
    wdithPos = widthPos.astype(int)

    dataPeak = np.zeros((nData,lenSignal),dtype=np.float32)  # peak
    baseData = np.empty((nData,lenSignal),dtype=np.float32)  # baseline

    xIndex = np.arange(lenSignal)

    iBins = nOrder - 1
    iBinWidthX = (lenSignal*3)//(iBins*4) # narrower than the actual bin width (lenSignal/iBin)
    nOrder_2 = nOrder-2

    numPeak = minPeak
    for i, (peakData,baseLine) in enumerate(zip(dataPeak,baseData)): 
        # generate a baseline using a spline curve
        baseX = np.linspace(0,lenSignal,nOrder) # generate nOrder points including 0 and lenSignal
        while  True:
            baseXt = baseX.copy()
            baseXt[1:-1] += rng.integers(-iBinWidthX,iBinWidthX, size=nOrder_2)
            if nOrder_2 == len(set(baseXt[1:-1])): # check if baseXt contains the same values
                baseX[:] = baseXt
                break

        cs = CubicSpline( np.sort(baseX), rng.random(nOrder) - 0.5 )
        #cs = Akima1DInterpolator(baseX, baseYt)
        baseLine[:] = cs(xIndex)

        # width를  균일하게 분포하도록 함 
        #peakWidth = rng.integers(minPeakWidth, maxPeakWidth, size=numPeak, endpoint=True)
        peakWidth = np.empty(numPeak, dtype=int) 
        for k in range(numPeak):
            binNum = (i+k) % widthNum # 적당한 곳부터 돌아가면서 width를 만듦
            peakWidth[k] = rng.integers(widthPos[binNum], widthPos[binNum+1], size=1, endpoint=True)        

        # 피크 영역을 큰 부분과 작은 부분 두 가지로 설정
        rangeWidth = rng.integers(lenSignal*2//5, lenSignal*3//5)
        rangePos = rng.integers(lenSignal-rangeWidth)
        rangeA = np.arange(rangePos, rangePos+rangeWidth, 2)
        
        rangeWidth = rng.integers(lenSignal//5, lenSignal*2//5)
        rangePos = rng.integers(lenSignal-rangeWidth)
        rangeB = np.arange(rangePos, rangePos+rangeWidth, 2)
        
        rangeAB = np.unique(np.append(rangeA, rangeB))
        peakPos = rng.choice(rangeAB, size=numPeak, replace=False)

        if numPeak<=2: 
            print('error: numPeak should be greater than 2..')
            return
        
        # 피크 크기를 대중소 3개로 구분해서 설정
        numPeakA, numPeakB = rng.integers(1, numPeak//4,endpoint=True), rng.integers(1, numPeak//4,endpoint=True)
        numPeakC = numPeak - numPeakA - numPeakB
        
        peakHeight = rng.random(size=numPeakA)*(1-0.75) + 0.75
        peakHeight = np.append(peakHeight, rng.random(size=numPeakB)*(0.5-0.25) + 0.25)
        if peakMinHeight < 0.025: peakMinHeight=0.025
        peakHeight = np.append(peakHeight,rng.random(size=numPeakC)*(0.15-peakMinHeight) + peakMinHeight)
                      
        #peakDataX = np.zeros(lenSignal+maxPeakWidth,dtype=np.float32)
        for k, (pkPos, pkWidth) in enumerate(zip(peakPos, peakWidth)):
            if pkPos + pkWidth < lenSignal: continue
            peakPos[k] -= pkWidth*3//4 # 1/4 정도만 밖으로 나가게 조정..
        
        lenSignalX = np.max(peakPos + peakWidth)
        if lenSignalX<lenSignal: lenSignalX = lenSignal
        
        # 피크 최종 생성
        peakDataX = np.zeros(lenSignalX,dtype=np.float32)
        for pkHeight, pkWidth, pkPos in zip(peakHeight, peakWidth, peakPos):
            peakDataX[pkPos:pkPos+pkWidth] += pkHeight*np.hanning(pkWidth)
        
        #peakData[:] = peakDataX[maxPeakWidth//2:lenSignal+maxPeakWidth//2]
        peakData[:] = peakDataX[:lenSignal]
        
        if i%(nData//10) == 0: print(f'{i:7d} data are created..')

        numPeak += 1
        if numPeak>maxPeak: numPeak=minPeak
            
    # baseline normalization
    baseData -= baseData.min(axis=1, keepdims=True)
    baseData /= baseData.max(axis=1, keepdims=True)
    if not np.isfinite(baseData).all(): print(f'NaN error in baseData[{i}]')

    dataPeak -= dataPeak.min(axis=1, keepdims=True)
    dataPeak /= dataPeak.max(axis=1, keepdims=True)
    if not np.isfinite(dataPeak).all(): print(f'NaN error in dataPeak{i}]')
        
    print('writing data files..')
    np.save(dataPath+'Peak.npy',dataPeak,allow_pickle=True)
    np.save(dataPath+'Base.npy',baseData,allow_pickle=True)

    print('Data generated..')
    return dataPeak, baseData

def mixData(dataPeak, dataBase, baseRatioMin=0.1, baseRatioMax=0.8, noiseMax=0.01):
    """
    noiseMax = None (without noise), 0.01(common)
    """
    baseRatio = ( rng.random( (len(dataPeak),1) )*(baseRatioMax-baseRatioMin) + baseRatioMin )
    baseRatio = np.repeat(baseRatio, len(dataPeak.T), axis = 1)
    peakRatio = 1 - baseRatio
    
    mixData = dataPeak*peakRatio + dataBase*baseRatio
    
    if noiseMax is not None:
        sigmaN = rng.random( len(mixData), 1 )*noiseMax        
        mixDataN = np.random.normal(mixData, scale=rng.random(len(mixData),1)*noiseMax*peakRatio )
    
    zMin = mixData.min(axis=1, keepdims=True)
    mixData -= zMin
    zDelta = mixData.max(axis=1, keepdims=True)
    mixData /= zDelta
    
    dataPeak = dataPeak/zDelta
    if noiseMax is None: return mixData, dataPeak*peakRatio/zDelta
    
    zMin = mixDataN.min(axis=1, keepdims=True)
    mixDataN -= zMin
    zDelta = mixDataN.max(axis=1, keepdims=True)
    mixDataN /= zDelta
    
    dataPeak = dataPeak/zDelta
    return mixDataN, mixData, dataPeak*peakRatio/zDelta

def prepareData(dataPath='', readValidData=True, noiseMax=0.005, baseRatioMin=0.1, baseRatioMax=0.8, onlyPeak=False, trainRatio=0.8):
    
    print('reading data from files..') 
    dataPeak=np.load(dataPath+'Peak.npy',allow_pickle=True)
    baseData=np.load(dataPath+'Base.npy',allow_pickle=True)
    print('reading done..')

    dataSize = len(baseData)
    dataLen  = len(baseData.T)
    dataLenX = len(dataPeak.T)
    dataLenD = dataLenX - dataLen

    trainSize = int(dataSize * trainRatio)
    validSize=dataSize-trainSize

    trainX = dataPeak[:trainSize]
    trainY = baseData[:trainSize]
    #validX = dataPeak[trainSize:trainSize+validSize]
    #validY = baseData[trainSize:trainSize+validSize]
    #validXX = np.copy(dataPeak[trainSize:trainSize+validSize])
    validX  = dataPeak[trainSize:]
    validY  = baseData[trainSize:]
    validZ  = np.copy(validX)

    print(trainX.shape, trainY.shape)
    print(validX.shape, validY.shape, validZ.shape)
    
    if readValidData:
        print('reading validation data from files..') 
        validXX=np.load(dataPath+'ValidX.npy',allow_pickle=True)
        validYY=np.load(dataPath+'ValidY.npy',allow_pickle=True)
        validZ =np.load(dataPath+'ValidZ.npy',allow_pickle=True)
        
        print(f'All data are prepared.. {validXX.shape}, {validYY.shape}, {validZ.shape}')
        return trainX, trainY, validZ , validXX, validYY

    validXX = np.empty_like(validX[:,:dataLen])
    validYY = np.empty_like(validXX)

    #plt.plot(validX[7],'b',validYY[7][:dataLen],'r')
    validDataSize = len(validX)
    validNoiseMax = noiseMax

    rmseNoise = 0
    rmseNoiseX = 0
    
    sigmaArray = rng.random(size=(len(validX),1))*validNoiseMax
    for i, (X, XX, Y, YY, Z) in enumerate(zip(validX,validXX,validY,validYY,validZ)):
        validBaseRatio = rng.random()*(baseRatioMax-baseRatioMin)+baseRatioMin
        validPeakRatio = 1 - validBaseRatio
        # adjust to outYY*validBaseRatio
        baseType = rng.integers(4)
        if baseType == 0:
            YY[:] = Y
        elif baseType == 1:
            YY[:] = Y[::-1]
        elif baseType == 2:
            YY[:] = (1-Y)
        else:
            YY[:] = (1-Y[::-1])

        # left/right peak exchange
        XX[:] = X if baseType%2 else X[::-1]
        
        YY *= validBaseRatio        
        XX *= validPeakRatio
        
        Zt = XX if onlyPeak else XX + YY
        # add AWGN
        #Z[:] = Zt + rng.normal(scale=rng.random()*validNoiseMax, size=dataLen)
        Z[:]  = rng.normal(Zt, scale=sigmaArray[i]*validPeakRatio)
                
        minData, maxData = Z.min(), Z.max()
        deltaMinMax = maxData-minData
        # normalize inXX and outYY again
        #XX -= minData
        XX /= deltaMinMax

        YY -= minData
        YY /= deltaMinMax

        Z  -= minData
        Z  /= deltaMinMax

        # normailize Zt to compute rmseNoise
        Zt  -= Zt.min()
        Zt  /= Zt.max()        
        
        rmseNoise  += np.sqrt( np.sum( (Z - (XX + YY) )**2 ) / dataLen )

        if i%(validDataSize//5) == 0:
            print(f'{i:7d} data are created..')

    print(f'RMSE : {rmseNoise/validSize:.10f}')

    print('writing validation data files..')
    np.save(dataPath+'ValidX.npy',validXX,allow_pickle=True)
    np.save(dataPath+'ValidY.npy',validYY,allow_pickle=True)
    np.save(dataPath+'ValidZ.npy',validZ ,allow_pickle=True)

    #print(validYY[0,0],validXX[0,0],sum((validYY[0]-validXX[0])**2)**(1/2))
    print(f'All data are prepared.. {trainX.shape}, {trainY.shape}')
    
    return trainX, trainY, validZ, validXX, validYY

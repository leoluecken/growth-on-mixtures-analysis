import numpy as np

class BatchExperiment(object):
    '''
    classdocs
    '''

    def __init__(self, experimentID):
        '''
        Constructor
        '''
        self._experimentID = experimentID
        self._runIDs = []
        
        # Measured quantities
        self._tSpan = {}
        self._sSpan = {}
        self._cdwSpan = {}
        self._tdaSpan = {}
        self._odSpan = {}
        self._co2Span = {}
        self._o2Span = {}
        
        # OD derived estimation for CDW
        self._cdwodSpan = {}

        # tSpan for the specific rates
        self._tSpanRates = {}
        
        # Specific rates based on measured cdw
        self._sRates = {}
        self._cdwodRates = {}
        self._cdwRates = {}
        self._tdaRates = {}
        self._odRates = {}
        self._co2Rates = {}
        self._o2Rates = {}

        # Specific rates based on cdw estimated via OD
        self._sRatesOD = {}
        self._cdwodRatesOD = {}
        self._cdwRatesOD = {}
        self._tdaRatesOD = {}
        self._odRatesOD = {}
        self._co2RatesOD = {}
        self._o2RatesOD = {}
    
    
    def getID(self):
        return self._experimentID
        
    def getRunIDs(self):
        return sorted(self._tSpan.keys())
    
    def getSubstrates(self):
        runIDs = self.getRunIDs()
        if len(runIDs):
            return sorted(self._sSpan[runIDs[0]].keys())
        else:
            return []
        
    def addRun(self, rid, data):
        if rid in self._runIDs:
            raise ValueError("BatchExperiment '%s': Run '%s' already exists!"%(self.getID(), rid))
        self._runIDs.append(rid)
        self._tSpan[rid] = data["t"]
        self._sSpan[rid] = data["S"]
        self._cdwSpan[rid] = data["CDW"]
        if "CDWOD" in data:
            self._cdwodSpan[rid] = data["CDWOD"]
        if "TDA" in data:
            self._tdaSpan[rid] = data["TDA"]
        if "OD" in data:
            self._odSpan[rid] = data["OD"]
        if "CO2" in data:
            self._co2Span[rid] = data["CO2"]
        if "O2" in data:
            self._o2Span[rid] = data["O2"]
        
        self._imputeMissing(rid)
        self._calculateSpecificRates(rid)
    
    
    def _imputeMissing(self, rid):
        for s in self._sSpan[rid].keys():
            self._interpolateMissing(self._tSpan[rid], self._sSpan[rid][s])
        self._interpolateMissing(self._tSpan[rid], self._cdwSpan[rid])
        if rid in self._cdwodSpan:
            self._interpolateMissing(self._tSpan[rid], self._cdwodSpan[rid])
        if rid in self._tdaSpan:
            self._interpolateMissing(self._tSpan[rid], self._tdaSpan[rid])
        if rid in self._odSpan:
            self._interpolateMissing(self._tSpan[rid], self._odSpan[rid])
        if rid in self._co2Span:
            self._interpolateMissing(self._tSpan[rid], self._co2Span[rid])
        if rid in self._o2Span:
            self._interpolateMissing(self._tSpan[rid], self._o2Span[rid])
        
    
    def _interpolateMissing(self, t, v):
        assert(len(t)==len(v))
        # Fills nans in the measurements with linearly interpolated values
        iMax = len(v)
        iMin = 0
        # Don't fill leading and trailing values since they can't be interpolated
        while np.isnan(v[iMin]):
            iMin+=1
        while np.isnan(v[iMax-1]):
            iMax-=1
        i = iMin
        while i < iMax:
            if np.isnan(v[i]):
                j=i+1
                while np.isnan(v[j]): j+=1
                # Linear interpolation of all values between i and j
                while i < j:
                    theta = (t[i]-t[i-1])/(t[j]-t[i-1])
                    v[i] = (1-theta)*v[i-1] + theta*v[j]
                    print("Interpolated missing value for '%s' at t=%f: %f"%(self._experimentID, t[i], v[i]))
                    i+=1
            else:
                i+=1
    
    
    def _calculateSpecificRates(self, rid):
        self._tSpanRates[rid] = (self._tSpan[rid][1:] + self._tSpan[rid][:-1])*0.5
        cdwSpanRates = (self._cdwSpan[rid][1:] + self._cdwSpan[rid][:-1])*0.5
        if rid in self._cdwodSpan:
            cdwodSpanRates = (self._cdwodSpan[rid][1:] + self._cdwodSpan[rid][:-1])*0.5

        tIncr = self._tSpan[rid][1:] - self._tSpan[rid][:-1]
        specificFactor = tIncr*cdwSpanRates
        if rid in self._cdwodSpan:
            specificFactorOD = tIncr*cdwodSpanRates
                
        self._sRates[rid], self._sRatesOD[rid] = {}, {}
        for s in self._sSpan[rid].keys():
            self._sRates[rid][s] = (self._sSpan[rid][s][1:] - self._sSpan[rid][s][:-1])/specificFactor
            if rid in self._cdwodSpan:
                self._sRatesOD[rid][s] = (self._sSpan[rid][s][1:] - self._sSpan[rid][s][:-1])/specificFactorOD
        
        self._cdwRates[rid] = (self._cdwSpan[rid][1:] - self._cdwSpan[rid][:-1])/specificFactor
        if rid in self._cdwodSpan:
            self._cdwodRates[rid] = (self._cdwodSpan[rid][1:] - self._cdwodSpan[rid][:-1])/specificFactor
        if rid in self._tdaSpan:
            self._tdaRates[rid] = (self._tdaSpan[rid][1:] - self._tdaSpan[rid][:-1])/specificFactor
        if rid in self._odSpan:
            self._odRates[rid] = (self._odSpan[rid][1:] - self._odSpan[rid][:-1])/specificFactor
        if rid in self._co2Span:
            self._co2Rates[rid] = (self._co2Span[rid][1:] - self._co2Span[rid][:-1])/specificFactor
        if rid in self._o2Span:
            self._o2Rates[rid] = (self._o2Span[rid][1:] - self._o2Span[rid][:-1])/specificFactor
        if rid in self._cdwodSpan:
            self._cdwRatesOD[rid] = (self._cdwSpan[rid][1:] - self._cdwSpan[rid][:-1])/specificFactorOD
        
        if rid in self._cdwodSpan:
            self._cdwodRatesOD[rid] = (self._cdwodSpan[rid][1:] - self._cdwodSpan[rid][:-1])/specificFactorOD
        if rid in self._odSpan:
            self._tdaRatesOD[rid] = (self._tdaSpan[rid][1:] - self._tdaSpan[rid][:-1])/specificFactorOD
        if rid in self._odSpan:
            self._odRatesOD[rid] = (self._odSpan[rid][1:] - self._odSpan[rid][:-1])/specificFactorOD
        if rid in self._co2Span:
            self._co2RatesOD[rid] = (self._co2Span[rid][1:] - self._co2Span[rid][:-1])/specificFactorOD
        if rid in self._o2Span:
            self._o2RatesOD[rid] = (self._o2Span[rid][1:] - self._o2Span[rid][:-1])/specificFactorOD
        
    
        

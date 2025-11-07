import numpy as np
from scipy.optimize import bisect
import utils
from warnings import warn
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from argparse import Namespace
from collections import defaultdict


class DEBDynamics(object):
    def __init__(self, params):
        '''
        Constructor
        '''
        self._substrates = params["substrates"]
        self._substrateIx = dict([(s,i) for i,s in enumerate(self._substrates)])
        self._nSubs = len(self._substrates)
        
        self._mu , self._K, self._yE = np.zeros(self._nSubs), np.zeros(self._nSubs), np.zeros(self._nSubs)
        
        self.setUptakeParams(params)
        self.setGrowthParams(params)
        
        # Interpolting objects for external forcings
        self._iPolDS = {}
        self._iPolS = {}
        self._iPolV = {}
        self._activeRunID = None
        self._considerP = True
        
        # Whether to include interactions in uptake
        self._considerInteractions = False
        # interactions substrateIndex->([indices], [strengths]) 
        self._a = {}
        
        # Parameter noise may contain trajectories of random parameter variation
        self._parameter_perturbations = defaultdict(dict)
        
    
    def setParameterNoise(self, rid, parameter_noise, noise_rng, T):
        # Pre-calculate Ornstein-Uhlenbeck perturbation trajectories over T
        noisy_pars = sorted(parameter_noise.keys())
        OU_trajs = dict()
        t_trans = 100.0
        dt = 0.001
        for p in noisy_pars:
            theta, sigma = parameter_noise[p]["theta"], parameter_noise[p]["sigma"]
            if p == "a":
                n = sum([len(self._a.get(six,[])) for six in self._substrateIx.values()])
            elif p in ["mu", "K", "yE"]:
                n = len(self._substrates)
            else:
                n = 1
            OU_trajs[p] = []
            for i in range(n):
                traj = generate_OU_traj(dt=dt, t0=0.0, t1=t_trans, x0=0.0, theta=theta, sigma=sigma, noise_rng=noise_rng)
                x0 = traj.y[-1]
                OU_trajs[p].append(generate_OU_traj(dt=dt, t0=T[0], t1=T[1], x0=x0, theta=theta, sigma=sigma, noise_rng=noise_rng))
            x = OU_trajs[p][0].t
            y = np.array([traj.y for traj in OU_trajs[p]])
            OU_trajs[p] = CubicSpline(x, y.T)
        self._parameter_perturbations[rid] = OU_trajs
    
        
    def setUptakeParams(self, params):
        for s, p in params.items():
            if not s in self._substrates: continue
            six = self._substrateIx[s]
            if "mu" in p:
                # Maximal consumption rate
                self._mu[six] = p["mu"]
            if "K" in p:
                # Half-saturation constant
                self._K[six] = p["K"]
            if "yE" in p:
                # Energy yield per substrate
                self._yE[six] = p["yE"]
            if "a" in p:
                assert("clusterIndices" in p)
                self._a[six] = (p["clusterIndices"], p["a"]) 
            
            
    def setGrowthParams(self, params):
        # Biomass yield per unit energy
        self._yV = params["yV"]
        # Mobilization rate of energy reserve
        self._rE = params["rE"]
        # Specific maintenance cost (in energy)
        self._m =  params["m"]
        if "yP" in params:
            # TDA yield per unit energy
            self._yP = params["yP"]
            # Decay rate of product
            self._rP = params["rP"]
        
        
    def getGrowthParams(self):
        params = {}
        params["yV"] = self._yV
        params["yP"] = self._yP
        params["rE"] = self._rE
        params["rP"] = self._rP
        params["m"]  = self._m
        return params
       
       
    def getUptakeParams(self, sid=None):
        params = {}
        if sid is None:
            params["mu"] = self._mu
            params["K"] = self._K
            params["yE"] = self._yE
        else:
            six = self._substrateIx[sid]
            params["mu"] = self._mu[six]
            params["K"] = self._K[six]
            params["yE"] = self._yE[six]

        return params
        
        
    def setDSInterpolation(self, rid, iPolDS):
        self._iPolDS[rid] = iPolDS
        
    def setSInterpolation(self, rid, iPolS):
        self._iPolS[rid] = iPolS
    
    def setVInterpolation(self, rid, iPolV):
        self._iPolV[rid] = iPolV
        
    def setActiveRunID(self, rid):
        self._activeRunID=rid
        
    def activateP(self):
        self._considerP = True
    def deactivateP(self):
        self._considerP = False
        
    def activateInteractions(self):
        self._considerInteractions = True
    def deactivateInteractions(self):
        self._considerInteractions = False
        
    def resetInteractions(self):
        self._a = {}
        
    def setInteractions(self, interactions):
        # interactions is a map (substrateID -> (a, clusterIndices))
        for sid, ints in interactions.items():
            six = self._substrateIx[sid]
            self._a[six] = []
            if type(ints) == list: 
                for i in ints:
                    if type(i) is dict:
                        self._a[six].append((i["clusterIndices"], i["a"]))
                    else:
                        self._a[six].append(i)
            else:
                if type(ints["a"]) == list:
                    # ensures back compatibility
                    for a, cli in zip(ints["clusterIndices"], ints["a"]):
                        self._a[six].append((a, cli))
                else:
                    self._a[six].append((ints["clusterIndices"], ints["a"]))
        print("post setInteractions(), a =", self._a)
            
    def getInteractions(self):
        # interactions is a map (substrateID -> (a, clusterIndices))
        print("getInteractions(), a =", self._a)
        interactions = {}
        for six, ints in self._a.items():
            interactions[self._substrates[six]] = [{"clusterIndices":i[0], "a":i[1]} for i in ints]
        return interactions
                
    def interactionsActive(self):
        return self._considerInteractions
        
    def _interactions(self, s, substrateIndex=None, param_perturbations=None):
        int_count = 0
        if substrateIndex is None:
            # return all interactions
            interactionTerms = np.zeros(len(s))
            if self._considerInteractions:
                # for six, ints in self._a.items():
                for six in sorted(self._substrateIx.values()):
                    if six in self._a:
                        ints = self._a[six]
                        for interaction in ints:
                            cli, ai = interaction
                            if param_perturbations is not None:
                                ai = np.maximum(0.0, ai + param_perturbations[int_count])
                                int_count += 1
                            interactionTerms[six] += sum([s[j] for j in cli])*ai
            return interactionTerms 
        else:
            # Return specific interaction term only
            sixs = sorted(self._substrateIx.values())
            six = sixs[0]
            while six != substrateIndex:
                int_count += len(self._a.get(six, []))
                six += 1
            self._a.setdefault(six,[])
            ints = self._a[substrateIndex]
            interactionTerm = 0.0
            if ints:
                # This is just messy - some parts use tuple of lists, others list of tuples... 
                if type(ints) == tuple:
                    iterints = zip(ints[0], ints[1])
                else:
                    iterints = ints
                for cli, ai in iterints:
                    if param_perturbations is not None:
                        ai = np.maximum(ai + param_perturbations[int_count], 0.0)
                        int_count += 1
                    try:
                        interactionTerm += np.sum([s[j] for j in cli])*ai
                    except Exception as e:
                        print("_interactions()\n  s=%s\n  six=%d"%(s, substrateIndex))
                        print("  cli:", cli)
                        print("  ai: ", ai)
                        raise e
            return interactionTerm 
        
    def __call__(self, t, X):
        # State variables:
        #  v - structural biomass
        #  e - energy reserve
        #  p - secondary metabolite production
        #  s - substrate concentration
        if self._considerP:
            v, e, p, s = X[0], X[1], X[2], np.maximum(X[3:], 0.0)
        else:
            v, e, s = X[0], X[1], np.maximum(X[2:],0.0)
        
        rid = self._activeRunID
        int_perturbations = (
            None 
            if self._parameter_perturbations.get(rid, None) is None else
            self._parameter_perturbations[rid].get("a", None)
            )
        if int_perturbations is not None:
            int_perturbations = int_perturbations(t)
        interactionTerms = self._interactions(s, param_perturbations=int_perturbations)
        
        K, mu, m = self._K, self._mu, self._m
        rE, yV, yE = self._rE, self._yV, self._yE
        if self._parameter_perturbations[rid]:
            # Add parameter perturbation trajectory to the value
            for par, traj in self._parameter_perturbations[rid].items():
                if par == "mu":
                    mu = np.maximum(0.0, mu + traj(t))
                elif par == "m":
                    m = np.maximum(0.0, m + traj(t)[0])
                elif par == "K":
                    K = np.maximum(0.0, K + traj(t))
                elif par == "rE":
                    rE = np.maximum(0.0, rE + traj(t)[0])
                elif par == "yE":
                    yE = np.maximum(0.0, yE + traj(t))
                elif par == "yV":
                    yV = np.maximum(0.0, yV + traj(t)[0])
                    
        if np.any(K):
            ds  = -mu*v*s/(K + s + interactionTerms)
        else:
            # If K=0, use an on/off switch for the dynamics
            if np.any(interactionTerms):
                warn("Using on/off feeding response is not integrated with interactions, currently. Ignoring interactions...")
            ds  = [-mu*v if ss > 0 else 0.0 for ss in s]
        
        de = -min(0.0, np.sum(ds*yE)) - e*rE
        dv  = (e*rE - v*m)*yV
            
        if self._considerP:
            rP, yP = self._rP, self._yP
            dp  = (e*rE - v*m)*yP - p*rP
            return np.hstack((dv, de, dp, ds))
        else:
            return np.hstack((dv, de, ds))
        
        
        
    def growthWithGivenSubstrate(self, t, X):
        if self._activeRunID not in self._iPolS:
            raise Exception("Set interpolation object (_iPolS) for substrate concentration first.")
        
        tdaFitting = len(X) == 3
        if tdaFitting:
            v, e, p = X
        else:
            v, e = X
        # Driven by S
        s = np.array([f(t) for f in self._iPolS[self._activeRunID]])
        interactionTerms = self._interactions(s)
        #print("interactionTerms:",interactionTerms)
        if np.any(self._K):
            ds  = -self._mu*v*s/(self._K + s + interactionTerms)
        else:
            # If K=0, use an on/off switch for the dynamics
            if np.any(interactionTerms):
                warn("Using on/off feeding response is not integrated with interactions, currently. Ignoring interactions...")
            ds  = [-self._mu*v if ss > 0 else 0.0 for ss in s]
        
        
        de = -min(0.0, np.sum(ds*self._yE)) - e*self._rE
        dv  = (e*self._rE - v*self._m)*self._yV
        if tdaFitting:
            dp  = (e*self._rE - v*self._m)*self._yP - p*self._rP
            dX = (dv, de, dp)
        else:
            dX = (dv, de)
        return dX

        
    def uptakeWithGivenBiomass(self, t, s, substrateIndex):
        rid = self._activeRunID
        if rid not in self._iPolV:
            raise Exception("Set interpolation object for biomass first.")
        v = self._iPolV[self._activeRunID](t)

        if self._considerInteractions:
            S = np.array([self._iPolS[self._activeRunID][six](t) for six in range(len(self._substrates))])
            interactionTerm = self._interactions(S, substrateIndex) 
        else:
            interactionTerm = 0.0
        
        yE, K, mu = self._yE, self._K, self._mu
        if self._parameter_perturbations[rid]:
            # Add parameter perturbation trajectory to the value
            for par, traj in self._parameter_perturbations[rid].items():
                if par == "mu":
                    mu = np.maximum(0.0, mu + traj(t))
                elif par == "K" and K is not None:
                    K = np.maximum(0.0001, K + traj(t))
                elif par == "yE":
                    yE = np.maximum(0.0, yE + traj(t))
            
        if self._K is not None:
            ds  = -mu[substrateIndex]*v*s/(K[substrateIndex] + s + interactionTerm)
        else:
            # If K=0, use an on/off switch for the dynamics
            if interactionTerm:
                warn("Using on/off feeding response is not integrated with interactions, currently. Ignoring interactions...")
            ds  = -mu[substrateIndex]*v if s > 0 else 0.0
        return ds
        
        
        
    def getStationaryReserve(self, V0, S0, DS0=None):
        '''
        1) If DS0 is not given, assume that the system is autonomous, i.e. all variables are
           dynamical. Calculate E0, such that d[E]/dt = 0, where [E]=E/V
              0 = d[E]/dt = (E'V-V'E)/V^2
          <=> E'V = V'E
          <=> (-S'*yE - E0*rE)*V0 =  E0*(E0*rE - V0*m)*yV
          <=> (yE*mu*V0*S0/(K + S0) - E0*rE)*V0 =  E0*(E0*rE - V0*m)*yV
          <=> yE*mu*V0*S0*V0/(K + S0) - E0*rE*V0 =  E0*E0*rE*yV - E0*V0*m*yV
          <=> 0 = rE*yV*E0*E0 - E0*(V0*m*yV + rE*V0) - yE*mu*V0*S0*V0/(K + S0)
          <=> 0 = E0*E0 - E0*(V0*m*yV + rE*V0)/rE*yV - yE*mu*V0*S0*V0/((K + S0)*(rE*yV)) =: Chi(E0) 
          
          For V0>0: Chi(0)<0 => exactly one root E0>0
          
        2) If DS0 is given, it indicates that S(t) is to be considered as an external forcing.
           In this case, S0 can be ignored and the substrate uptake rate is taken to be DS0 
        '''
        if V0 == 0.0:
            return 0.0
        
        if DS0 is None:
            return self._getStationaryReserveFullyDynamical(V0, S0)
        else:
            return self._getStationaryReserveDriven(V0, DS0)
        
    def _getStationaryReserveFullyDynamical(self, V0, S0):
        
        # calculate uptake rates for all substrates
        interactionTerms = self._interactions(S0)
        if np.any(self._K):
            uptakeRates  = -self._mu*V0*S0/(self._K + S0 + interactionTerms)
        else:
            # If K=0, use an on/off switch for the dynamics
            if np.any(interactionTerms):
                warn("Using on/off feeding response is not integrated with interactions, currently. Ignoring interactions...")
            uptakeRates  = [-self._mu*V0 if ss > 0 else 0.0 for ss in S0]
        
        # Polynomial coefficients
        p = np.ones(3)
        p[1] = -(self._m/self._rE - 1/self._yV)*V0
        p[2] = V0*V0*np.dot(self._yE, uptakeRates)/(self._yV*self._rE)
        
        roots = sorted(np.roots(p))
        #print("roots:", roots)
        if(roots[0]*roots[1] >= 0):
            raise Exception("stationary E/V should be the unique, positive solution of the above polynomial!!!")
        E0 = roots[1]
        return E0
    
        
    def _getStationaryReserveFullyDynamicalOld(self, V0, S0):
#         # p-q formula
#         p2 = 0.5*(V0*self._m*self._yV + self._rE*V0)/(self._rE*self._yV)
#         q = -self._yE*self._mu*V0*S0*V0/((self._K + S0)*(self._rE*self._yV))
#         E0 = -p2 + np.sqrt(p2*p2 - q)
        
        # Solve generically with find_zero
        pOn = self._considerP
        self.deactivateP()
        def f(e):
            D = self.__call__(0.0, np.hstack((V0, e, S0)))
            dv, de = D[0], D[1]
            return (de*V0 - dv*e)
        
        
        success = False
        factor = 10; maxFactor = 1000000
        while (not success) and (factor<=maxFactor):
            try:
                E0 = bisect(f, 0.0, max(V0,0.001)*factor)
                success = True
            except:
                #print("Bisect() failed with (V0=%.3f, S0=%.3f)"%(V0,S0))
                factor*=10
        
        if not success:
            res = minimize(f, max(V0,0.001)*50)
            E0 = res.x
        if pOn: self.activateP()
#         print("E0:",E0)
        return E0
    
    
    def _getStationaryReserveDriven(self, V0, DS0):
        # Solve generically with find_zero        
        def f(e):
            # return -min(0.0, np.sum(DS0*self._yE)) - e*self._rE
            return self.growthWithGivenUptake(0.0, [V0, e, 0.0])[1]

        success = False
        factor = 10; maxFactor = 1000000
        while (not success) and (factor<=maxFactor):
            try:
                E0 = bisect(f, 0.0, max(V0,0.001)*factor)
                success = True
            except:
                #print("Bisect() failed with (V0=%.3f, DS0=%.3f)"%(V0,DS0))
                factor*=10
                                
        if not success:
            import matplotlib.pyplot as plt
            print("Bisect() failed with (V0=%.3f, DS0=%s)"%(V0,DS0))
            eSpan=np.linspace(0.0, max(V0,0.001)*factor, 101)
            print(eSpan)
            print(self._yE)
            print(self._yV)
            plt.plot(eSpan, [f(e) for e in eSpan])
            plt.title("V0=%s, DS0=%s, success=%s"%(V0, DS0, success))
            plt.show()
        
        if not success:
            raise Exception("Bisect() failed with (V0=%.3f, DS0=%s)"%(V0,DS0))
        
        return E0
        
        
def generate_OU_traj(dt, t0, t1, x0, theta, sigma, noise_rng):
    def step(dt, x0, theta, sigma):
        r = noise_rng.normal()
        x1 = x0 - theta*x0*dt + sigma*np.sqrt(dt)*r
        return x1
    
    def integrate(dt, t0, t1, x0, theta, sigma):
        tspan = [t0]
        xspan = [x0]
        while tspan[-1] < t1:
            x1 = step(dt, xspan[-1], theta, sigma)
            tspan.append(tspan[-1] + dt)
            xspan.append(x1)
        return tspan, xspan
    
    tspan, xspan = integrate(dt, t0, t1, x0, theta, sigma)
    traj = Namespace(t=tspan, y=xspan)
    return traj

'''
This implements simulations and plotting for a simple toy setup to
illustrate the multisubstrate DEB model with interactions
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import fitting, plotting


class MockExp(object):
    def __init__(self, substrates, S0, tend, runID):
        N = len(substrates)
        self._runIDs = []
        self._substrates = substrates
        self._tSpan = {runID:[0.0, tend]}
        self._sSpan = {runID:dict([(s, [S0[i]]) for i,s in enumerate(substrates)])}
        

def makeParamsDict(expID, substrates, mu, K, yE, yV, rE, m):
    params = {}
    params["substrates"] = substrates
    params["V0"] = {expID : 0.1}
    for i, s in enumerate(substrates):
        params[s] = {}
        params[s]["mu"] = mu[i]
        params[s]["K"]  = K[i]
        params[s]["yE"] = yE[i]
    
    # Growth params
    params["yV"] = yV
    params["rE"] = rE
    params["m"]  = m

    return params


def calculateGrowthScore(sol):
    ''' Calculates the average biomass concentration
        during the experiment: integral(V(t))/T
    '''
    dt = 0.1
    Nt = (sol.t[-1]-sol.t[0])/dt
    dt = dt*int(Nt)/Nt
    Nt = int(Nt)
    T = np.linspace(sol.t[0], sol.t[-1], Nt+1)
    V = sol.sol(T)[0]
    I = np.sum(V[1:]+V[:-1])*0.5*dt/sol.t[-1]
    
    print("calculateGrowthScore()")
    print("dt =",dt)
    print("Nt =",Nt)
    print("Nt*dt =",Nt*dt)
    print("I =",I)
    return(I)
    

def run():
    # Uptake params per substrate
    substrates = ["A", "B"]
    N = len(substrates)
    # Inhibited substrates
    inhibited  = {
        "both" : ["A", "B"],
        "none" : [],
        "A"    : ["A"],
        "B"    : ["B"]
        }
    # Initial concentrations
    S0         = [7.0, 6.0]  
    
    mu         = [0.5, 1.0]
    K          = [2.5, 1.5]
    yE         = [30.0, 40.0]
    
    # Growth params
    yV = 0.002
    rE = 0.35
    m  = 0.05
        
    # Interactions between substrates
    # Interactions is a map (substrateID -> (a, clusterIndices)), therefore, we 
    # use the peculiar form of specifying interactions between singleton clusters
    # and single substrates.
    clusters = dict([(s,[i]) for i,s in enumerate(substrates)])
    interactions = {
        "A": {"a":50.0, "clusterIndices":clusters["B"]},
        "B": {"a":10.0, "clusterIndices":clusters["A"]}
        }
    
    # Simulation interval duration
    tend = 150
        
    # Experiment title
    expID = "test"
    
    # Prepare params
    params = makeParamsDict(expID, substrates, mu, K, yE, yV, rE, m)
    
    # Set up the DEB fitter instance, which provides ODE solutions    
    fitter = fitting.DEBFitter(MockExp(substrates, S0, tend, expID), params)
    fitter.setClusters(clusters)
    fitter.activateInteractions()
    
    growthScore = {}
    
    for ID, inh in inhibited.items():
        ints = dict([(k,v) for k,v in interactions.items() if k in inh])
        fitter.resetInteractions()
        fitter.setInteractions(ints)
        sol, rhs = fitter.generateFullSolution(expID, False)
        growthScore[ID] = calculateGrowthScore(sol)
        
        # Plot the ODE solution (concentrations)
        fig, axes = plt.subplots(2, 1)
        print("axes:",axes)
        ax = axes[0]
        plotting.plotSubstrateCurveSim(sol, np.arange(N)+2, substrates, ax, False, False, params)
        ax.set_ylim((0, 8.0))
        ax.set_ylabel("$S$")
        ax2 = plt.twinx(ax)
        plotting.plotGrowthCurveSim(sol, False, ax2, False)
        ax2.set_ylabel("$V$")
        ax2.set_ylim((0, 2.5))
        
        plotting.align_yaxis(ax, ax2)
        handles, labels = ax.get_legend_handles_labels()
        labels = ["$"+l+"$" for l in labels]
        handles2, labels2 = ax2.get_legend_handles_labels()
        labels2 = ["$V$" if l=="CDW" else "$"+l+"$" for l in labels2]
        ax2.legend(handles=handles+handles2, labels=labels+labels2, loc="best")

        # Plot the ODE solution (rates)
        #fig, ax = plt.subplots()
        ax = axes[1]
        plotting.plotSubstrateRateSim(sol, rhs, np.arange(N)+2, substrates, ax, False, None)
        ax.set_ylim((0, 1.0))
        ax.set_ylabel("$[\\dot{S}]$")
        ax2 = plt.twinx(ax)
        plotting.plotGrowthRateSim(sol, rhs, plotTDA=False, ax=ax2)
        ax2.set_ylim((0, 0.1))
        ax2.set_ylabel("$[\\dot{V}]$")
        plotting.align_yaxis(ax, ax2)
        handles, labels = ax.get_legend_handles_labels()
        labels = [("$"+l+"$").replace("_","") for l in labels]
        handles2, labels2 = ax2.get_legend_handles_labels()
        labels2 = ["$[dV/dt]$" if l=="specific growth rate" else "$"+l+"$" for l in labels2]
        ax2.legend(handles=handles+handles2, labels=labels+labels2, loc="best")

        title = "Interactions: "
        for s, d in ints.items():
            title += substrates[d["clusterIndices"][0]] + "->" + str(d["a"]) + "->" + s + " "
        figname = "illustrativeSetup_inhibited_"+ID
        figname = os.path.join(plotting.FIG_DIR, figname+".svg")
        plt.title(title)    
        fig.set_size_inches((4,3))
        plt.tight_layout()
        plt.savefig(figname)
        print(f"Saved fig '{figname}")

        # plt.show()
        
    print("Growth scores:")
    for k,v in growthScore.items():
        print(k,v)
    


if __name__ == '__main__':
    run()
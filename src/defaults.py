import config

growthParams = {}

# Optimized
growthParams["yV"] = 0.00165134
growthParams["yP"] = 0.00373625
growthParams["rE"] = 0.26590382
growthParams["rP"] = 0.06393319
growthParams["m"]  = 0.09471784

uptakeParams = {}
uptakeParams["K"]  = 1.0
uptakeParams["mu"] = 1.0
uptakeParams["a_large"] = 1.0
uptakeParams["a_small"] = 0.1

_defaultV0 = 0.015


# Parameter distributions
param_dists = dict(
    # Yield V/E
    yV = dict(type="norm", loc=0.0016, scale=0.0005),
    # Rate E→V 
    rE = dict(type="norm", loc=0.265, scale=0.05),
    # Decay V
    m  = dict(type="norm", loc=0.095, scale=0.025),
    # Half saturation consts
    K  = dict(type="uniform", loc=0.001, scale=2.0),
    # Max uptake rates
    mu  = dict(type="norm", loc=0.5, scale=0.5),
    # Yield coeffs E/S
    yE  = dict(type="norm", loc=40.0, scale=5.0),
    # Inhibition coefficients
    a   = dict(type="uniform", loc=0.0, scale=50),
    
    # Initial biomass density
    V0 = dict(type="norm", loc=0.02, scale=0.01),
    # Initial substrate concentrations,
    S0 = dict(type="norm", loc=1.35, scale=0.1),
    )


def default_for_sampled_params(nRuns, substrates):
    defaultV0 = param_dists["V0"]["loc"]
    params = dict(
        yV = param_dists["yV"]["loc"],
        rE = param_dists["rE"]["loc"],
        m  = param_dists["m"]["loc"],
        V0 = dict([("F"+str(i+1), defaultV0) for i in range(nRuns)])
        )
    for s in substrates:
        params[s] = {}
        params[s]["yE"] = param_dists["yE"]["loc"]
        params[s]["K"]  = param_dists["K"]["loc"]
        params[s]["mu"] = param_dists["mu"]["loc"]
    params["substrates"] = substrates
    return params
    


def makeParams(substrates, nRuns, synthetic=False):
    if synthetic:
        return default_for_sampled_params(nRuns, substrates)
    
    params = growthParams.copy()
    params["V0"] = dict([("F"+str(i+1), _defaultV0) for i in range(nRuns)])
    for s in substrates:
        params[s] = {}
        params[s]["yE"] = config.ATPYields[s]
        params[s]["K"]  = uptakeParams["K"]
        params[s]["mu"] = uptakeParams["mu"]
    params["substrates"] = substrates
    return params

import os
from copy import deepcopy
from pprint import pp
from numpy.random import default_rng
from scipy.stats import norm, uniform
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.dag import is_directed_acyclic_graph
from pyvis.network import Network
from collections import defaultdict

import fitting
from main_model_illustration import MockExp
import plotting
from argparse import Namespace
import batchexperiment
from config import PROJECT_DIR, FIG_DIR, substrate_colors, hash_color
import tomllib
import toml
from defaults import param_dists

synth_traj_figdir = FIG_DIR / "synthetic_trajs"
if not synth_traj_figdir.exists():
    os.makedirs(synth_traj_figdir, exist_ok=True)
    

# Parameter for Ornstein-Uhlenbeck processes governing the 
# dynamic parameter perturbations
#     dX = -theta*X*dt + sigma*dW
# We integrate this by the Euler-Maruyama method:
#     X(t+dt) = X(t) - theta*X(t)*dt + sigma*sqrt(dt)*Z,
# with Z ~ N(0,1)
dynamic_noise = dict(
    mu = dict(theta=0.1, sigma=0.1),
    # m  = dict(theta=0.1, sigma=0.01),
    # rE = dict(theta=0.1, sigma=0.01),
    # a = dict(theta=0.1, sigma=1.0),
    )

measurement_noise_varc = dict(
    CDW = 0.1,
    subs = 0.1,
    )

minvals = dict(
    yV = 0.00001,
    rE = 0.01,
    m  = 0.001,
    K  = 0.01,
    mu  = 0.01,
    yE  = 0.5,
    a   = 0.001,
    V0 = 0.0001,
    S0 = 0.1,
    )




def sample_pdist(distspec, minval, size, rng):
    if distspec["type"] == "norm":
        dist = norm
    elif distspec["type"] == "uniform":
        dist = uniform
    else:
        raise Exception(f"Unknown dist type '{distspec['type']}'")

    size = 1 if size is None else size
    fails, maxfails = 0, 250*size
    samples = []
    while len(samples) < size and fails < maxfails:
        sample = dist.rvs(loc=distspec["loc"], scale=distspec["scale"], random_state=rng)
        if sample >= minval:
            samples.append(sample)
        else:
            fails += 1
    if fails == maxfails:
        raise Exception(f"Couldn't find feasible sample > {minval:g} for distspec: {distspec}")
            
    if size == 1:
        return samples[0]
    return samples
    
    
def randomInteractions(substrates, total_interaction_nr, rng):
    interactions = dict()

    g = nx.DiGraph()
    for s in substrates:
        g.add_node(s)
        
    i = 0
    fails, maxfails = 0, 250
    while i < total_interaction_nr and fails < maxfails:
        # Inhibited substrate
        ix0 = rng.integers(len(substrates))
        s0 = substrates[ix0]
        # inhibiting substrate
        present_ints = [d["clusterIndices"][0] for d in interactions.get(s0,[])]
        candidates = set(range(len(substrates))).difference(set(present_ints+ix0))
        while len(candidates) > 0:
            ix1 = rng.choice(sorted(candidates))
            s1 = substrates[ix1]
            if (s1, s0) in g.edges():
                candidates.remove(ix1)
                continue
            g.add_edge(s1, s0)
            if is_directed_acyclic_graph(g):
                break
            else:
                g.remove_edge(s1, s0)
                candidates.remove(ix1)
        if len(candidates) == 0:
            fails+=1
            continue
        if fails == maxfails:
            raise Exception(f"Failed constructing {total_interaction_nr} interactions without constructing cyclic graph.")
        a = sample_pdist(param_dists["a"], minvals["a"], size=None, rng=rng)
        interactions.setdefault(s0, [])
        interactions[s0].append({"a":a, "clusterIndices":[ix1]})
        i+=1 
        
    pp(interactions)
    # # Interactions
    # interactions = {
    #     "A": [{"a":50.0, "clusterIndices":[1]}],
    #     "B": [{"a":10.0, "clusterIndices":[0]}],
    #     }
    return interactions


def randomParams(substrates, runIDs, total_interaction_nr, seed):
    # Randomize all parameters for the different substrates
    # and the general growth parameters. 
    rng = default_rng(seed)
    nsubs = len(substrates)
    # Growth params
    params = dict(substrates=substrates, seed=seed) 
    for k in ["m", "rE", "yV"]:
        params[k] = sample_pdist(param_dists[k], minvals[k], size=1, rng=rng)
    # Substrate params
    subpars = {}
    for k in ["K", "yE", "mu"]:
        subpars[k] = dict(zip(substrates, sample_pdist(param_dists[k], minvals[k], size=nsubs, rng=rng)))
    for s in substrates:
        params[s] = {k:d[s] for k, d in subpars.items()}
    # Interactions
    interactions = randomInteractions(substrates, total_interaction_nr, rng)
    params["interactions"] = interactions
    # Initial states
    params["V0"], params["S0"] = {}, {}    
    for rid in runIDs:
        params["V0"][rid] = sample_pdist(param_dists["V0"], minvals["V0"], size=1, rng=rng)
        params["S0"][rid] = dict(zip(substrates, sample_pdist(param_dists["S0"], minvals["S0"], 
                                                              size=nsubs, rng=rng)))
    return params


def plotTrajectory(substrates, sol, experiment, N, params, rhs, 
                   title, rid, outdir, 
                   ylim=(0.0, 8.0), figname=None, show=False):
    # Plot the ODE solution (concentrations)
    fig, axes = plt.subplots(nrows=2, layout="constrained", figsize=(5,4))
    print("axes:",axes)
    ax = axes[0]
    ax.set_title(title)
    plotting.plotSubstrateCurveSim(sol, np.arange(N)+2, substrates, ax, False, False, params)
    plotting.plotSubstrateCurveData(experiment, rid, substrates, ax, False, ls="x", logscale=False, 
                           clusterColor=None, labels=False)    
    ax.set_ylim(ylim)
    ax.set_ylabel("$S$")
    ax2 = plt.twinx(ax)
    plotting.plotGrowthCurveSim(sol, False, ax2, False)
    plotting.plotGrowthCurveData(experiment, rid, False, ax2, 
                        biomassIndicator="CDW", ls="x", labels=False)
    ax2.set_ylabel("$V$")
    # ax2.set_ylim((0, 2.5))
    
    plotting.align_yaxis(ax, ax2)
    handles, labels = ax.get_legend_handles_labels()
    labels = ["$"+l+"$" for l in labels]
    handles2, labels2 = ax2.get_legend_handles_labels()
    labels2 = ["$V$" if l=="CDW" else "$"+l+"$" for l in labels2]
    ax2.legend(handles=handles+handles2, labels=labels+labels2, 
               loc="best", ncols=2)
    ax.set_xlabel("$t$ [h]")

    # Plot the ODE solution (rates)
    #fig, ax = plt.subplots()
    ax = axes[1]
    plotting.plotSubstrateRateSim(sol, rhs, np.arange(N)+2, substrates, ax, False, None)
    plotting.plotSubstrateRateData(experiment, rid, substrates, ax, 
                                   aggregateSubstrates=False, ls="x", labels=False)
    ax.set_ylim((0, 1.0))
    ax.set_ylabel("$[\\dot{S}]$")
    ax2 = plt.twinx(ax)
    plotting.plotGrowthRateSim(sol, rhs, plotTDA=False, ax=ax2)
    plotting.plotGrowthRateData(experiment, rid, False, ax2, 
                                biomassIndicator="CDW", ls="x")
    # ax2.set_ylim((0, 0.1))
    ax2.set_ylabel("$[\\dot{V}]$")
    plotting.align_yaxis(ax, ax2)
    handles, labels = ax.get_legend_handles_labels()
    labels = [("$"+l+"$").replace("_","") for l in labels]
    handles2, labels2 = ax2.get_legend_handles_labels()
    labels2 = ["$[dV/dt]$" if l=="specific growth rate" else "$"+l+"$" for l in labels2]
    #ax2.legend(handles=handles+handles2, labels=labels+labels2, loc="best")
    ax2.legend().remove()
    ax.set_xlabel("$t$ [h]")
    
    if figname is None:
        figname = "random_system_%d"%params["seed"]
    figname = outdir / (figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plotParameterNoise(substrates, fitter, pars, rid, title, ylim=None, outdir=None, show=True):
    dyns = fitter._dynamics
    noise_trajs = dyns._parameter_perturbations[rid]
    #runIDs = dyns._parameter_perturbations
    # fig, axes = plt.subplots(nrows=int(np.ceil(np.round(len(runIDs)/2,1))), 
    #                          ncols=2, layout="constrained")
    fig, ax = plt.subplots(layout="constrained", figsize=(5,2.5))
    # axes = np.array([ax])
    
    tspan = fitter._experiment._tSpan[rid]
    tspan = np.arange(tspan[0], tspan[1], 0.25)
    
    # for s, ax in zip(substrates, axes.flatten()):
    for pn, trajs in noise_trajs.items():
        perts = trajs(tspan)
        for s in substrates:
            six = dyns._substrateIx[s]
            pv = pars[s][pn]
            col = hash_color(s+"asfda")
            ax.plot([tspan[0], tspan[-1]], [pv, pv], color=col, lw=0.5, ls="--", zorder=-1)
            ax.plot(tspan, np.maximum(0.0, pv + perts[:, six]), label=f"{pn} ({s})", color=col, lw=1.2)
    ax.plot([tspan[0], tspan[-1]], [0, 0], color="k", lw=0.5, ls="--", zorder=-2)
    ax.legend(ncols=2)
    ax.set_xlabel("$t$ [$h$]")
    ax.set_ylabel("$\mu_i$ [$h^{-1}$]")
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title(title)
    if outdir:
        figname = outdir / (title+".svg")
        fig.savefig(figname)
        print(f"Saved figure '{figname}'")
    if show:
        plt.show()
    else:
        plt.close("all")
        

def plotInhibitionNetwork(params):
    substrates = params["substrates"]
    interactions = params["interactions"]
    
    def nodename(s):
        return(s.split("(")[0].strip())
    
    nt = Network('500px', '500px', directed=True, notebook=False)
    for s in substrates:
        n = nodename(s)
        nt.add_node(n, size=np.sqrt(params[s]["mu"]/(params[s]["K"]+1))*10)
    for s in substrates:
        n = nodename(s)
        for i in interactions.get(s, []):
            a, ix = i["a"], i["clusterIndices"][0]
            n2 = nodename(substrates[ix])
            nt.add_edge(n2, n, width=np.sqrt(a), 
                        arrowStrikethrough=False, title="a=%g"%a)

    figname = "random_system_%d_nw"%params["seed"]
    figname = os.path.join(plotting.FIG_DIR, figname+".html")
    #nt.from_nx(g, edge_scaling=False, show_edge_weights=False)
    nt.show("test.html", notebook=False)
    
    
def observe_sln(sln, tspan, measurement_noise_varc, measurement_rng):
    # This adds an observation error to CDW and substrate concentrations
    # Given a coefficient of variation vc at x=1, an observation x is modified as 
    #    x_pert = max(0.0, x*(1 + Z)), 
    # where Z ~ N(0,vc*sqrt(x))
    solSpan = sln.sol(tspan) 
    subSpans = solSpan[2:] 
    varc = measurement_noise_varc["CDW"]
    CDW = solSpan[0]
    CDW_obs = np.zeros_like(tspan, dtype=float)
    for i, cdw in enumerate(CDW):
        cdw = max(cdw, 0.0)
        pert = measurement_rng.normal(0.0, scale=varc*np.sqrt(cdw))
        obs = max(cdw*(1 + pert), 0.0)
        CDW_obs[i] = obs
    varc = measurement_noise_varc["subs"]
    subSpans_obs = np.zeros_like(subSpans)
    for six, sspan in enumerate(subSpans):
        sspan_obs = np.zeros_like(sspan, dtype=float)
        for i, s in enumerate(sspan):
            s = max(s,0.0)
            pert = measurement_rng.normal(0.0, scale=varc*np.sqrt(s))
            obs = max(s*(1 + pert), 0.0)
            sspan_obs[i] = obs
        subSpans_obs[six,:] = sspan_obs
    observed_sln = Namespace(t = tspan, CDW = CDW_obs, substrates = subSpans_obs)
    return observed_sln
    
    
def make_experimental_data(observed_sln, substrates):
    # Create an experiment from a synthetic observation
    data = dict(
        t = observed_sln.t,
        S = {s:observed_sln.substrates[i,:] for i, s in enumerate(substrates)},
        CDW = observed_sln.CDW,
        CDWOD = observed_sln.CDW,
        )
    return data


def save_parameter(fn, pars):
    with open(fn, "w") as f:
        toml.dump(pars, f)
        print(f"Wrote params to '{fn}'")


def generate_synthetic_experiment(outdir,
    nruns, tend, tspan_obs, nsubs, total_interaction_nr,
    seed, dynamic_noise_intensity, measurement_noise_intensity,
    expID, 
    plot_trajs=True, plot_nw=False, rate_plot=False, save_params=True):
    
    rng = np.random.default_rng(seed)
    dynamic_noise_seed = rng.integers(0, 99999)
    measurement_noise_seed = rng.integers(0, 99999)
    
    measurement_noise = deepcopy(measurement_noise_varc)
    for k in measurement_noise:
        measurement_noise[k] *= measurement_noise_intensity
        
    if dynamic_noise_intensity:
        parameter_noise = deepcopy(dynamic_noise)
        for k in parameter_noise:
            parameter_noise[k]["theta"] /= dynamic_noise_intensity
    else:
        parameter_noise = None
    
    runIDs = ["F%d"%i for i in range(1,nruns+1)]
    substrates = [f"S{i:02d}" for i in range(nsubs)]
    pars = randomParams(substrates, runIDs, total_interaction_nr, seed)
    
    if plot_nw:
        plotInhibitionNetwork(pars)
    
    clusters = {s:[i] for i,s in enumerate(substrates)}
    # Use MockExp for generating the reference timeline
    noise_rng = default_rng(dynamic_noise_seed)
    measurement_rng = default_rng(measurement_noise_seed)
    experimental_data = batchexperiment.BatchExperiment(expID)
    for rid in runIDs:
        fitter = fitting.DEBFitter(MockExp(substrates, [pars["S0"][rid][s] for s in substrates], tend, rid), pars)
        fitter.setClusters(clusters)
        fitter.setInteractions(pars["interactions"])
        fitter.activateInteractions()
        sln, dyns = fitter.generateFullSolution(rid=rid, includeTDA=False, parameter_noise=parameter_noise, noise_rng=noise_rng)
        
        observed_sln = observe_sln(sln, tspan_obs, measurement_noise, measurement_rng)
        data = make_experimental_data(observed_sln, substrates)
        experimental_data.addRun(rid, data) 
            
        if rate_plot:
            title = "dynparams_seed%d (run %s, I_dyn=%g, I_obs=%g)"%(seed, rid, dynamic_noise_intensity, measurement_noise_intensity)
            debug_noise_ylim = (-0.05, 2.05)
            plotParameterNoise(substrates, fitter, pars, rid, title=title, outdir=outdir, ylim=debug_noise_ylim, show=False)
            
        if plot_trajs:
            title = "traj_seed%d (run %s, I_dyn=%g, I_obs=%g)"%(seed, rid, dynamic_noise_intensity, measurement_noise_intensity)
            plotTrajectory(substrates, sln, experimental_data, len(substrates), pars, dyns, title, rid, outdir,
                           figname=title, show=False, ylim=(0.0, 2.2))
    if save_params and (outdir is not None):
        fn = outdir / f"pars_{expID}.toml"
        save_parameter(fn, pars)
    return experimental_data
        
            
def save_experiment_data(exp, outdir):
    fn = "data_" + exp._experimentID + ".csv"
    fn = outdir / fn
    
    data = defaultdict(list)
    runIDs = sorted(exp._tSpan)
     
    for rid in runIDs:
        data["CDW"].extend(exp._cdwSpan[rid])
        data["CDWOD"].extend(exp._cdwodSpan[rid])
        data["t"].extend(exp._tSpan[rid])
        sSpans = exp._sSpan[rid]
        for s, sSpan in sSpans.items():
            sname = s.split("(")[0].strip()
            data[sname].extend(sSpan)
        data["rid"].extend([rid]*len(exp._tSpan[rid]))
    
    df = pd.DataFrame(data)
    df.to_csv(fn)
    print(f"Saved experimental data to {fn}")
    return fn


def test():
    nruns = 4
    tend = 150
    # Time points for the observed data
    tspan_obs = np.arange(0,tend,4)
    nsubs = 10
    total_interaction_nr = 10
    seed = 2325
    # Factor to control the total intensity of the dynamic noise
    dynamic_noise_intensity = 0.1
    # Factor to control the total intensity of the observation noise
    measurement_noise_intensity = 0.1
    
    outdir = synth_traj_figdir
    
    exp = generate_synthetic_experiment(outdir=outdir,
        nruns=nruns, tend=tend, tspan_obs=tspan_obs, 
        nsubs=nsubs, total_interaction_nr=total_interaction_nr,
        seed=seed, dynamic_noise_intensity=dynamic_noise_intensity, 
        measurement_noise_intensity=measurement_noise_intensity,
        expID="test",
        plot_trajs = True, plot_nw = False, debug_plot=True, 
        save_params = False
        )
    
    save_experiment_data(exp, outdir)
    
    
if __name__=="__main__":
    test()







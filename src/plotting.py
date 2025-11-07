from collections import defaultdict
import os
from pathlib import Path
from pprint import pp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_rgb
import pandas as pd
import seaborn as sns
from scipy.interpolate.interpolate import interp1d
import fitting
from functools import reduce
import utils
from config import FIG_DIR, subs_long_names

import config
from utils import align_yaxis


plotDimensions = {}
plotDimensions["timeline"] = (6,3)
plotDimensions["clusterTimeline"] = (6,3)
plotDimensions["timeline_wide"] = (8,2.5)
plotDimensions["timeline_growth"] = (8,4)
plotDimensions["table"]    = (10,3)
plotDimensions["matrix"]   = [3., 2.7]
plotDimensions["table_small"]    = (3.5,2.5)


def plotExperiment(multiSubstrateExperiment, show=True):
    nrows=6; ncols=2
    ylim_subs = [-0.1, 1.8]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, 
                             layout="constrained", figsize=(8, 10))
    runIDs = multiSubstrateExperiment.getRunIDs()
    substrates = multiSubstrateExperiment.getSubstrates()
    
    axes_subs = axes.flatten()[1:]
    ax_cdw = axes.flatten()[0]
    
    for run, rid in enumerate(runIDs):
        run_color = config.run_colors[rid]
        t = multiSubstrateExperiment._tSpan[rid]
        cdw = multiSubstrateExperiment._cdwodSpan[rid]
        sns.lineplot(x=t, y=cdw, label=f"replicate {run+1}",
                        marker="x", ms=5, mec=None, mfc=run_color,  
                        ls="--", lw=0.8, zorder=0,                        
                        color=run_color, ax=ax_cdw)
        ax_cdw.set_ylabel("CDW [g/l]")
        for subs, ax in zip(substrates, axes_subs):
            s = multiSubstrateExperiment._sSpan[rid][subs]
            sns.lineplot(x=t, y=s, label=f"replicate {run+1}",
                            marker="x", ms=5, mec=None, mfc=run_color,  
                            ls="--", lw=0.8, zorder=0,                        
                            color=run_color, ax=ax)
            ax.set_ylim(ylim_subs)
            ax.set_ylabel(f"mM {subs}")
            ax.legend().remove()
    xlim = ax_cdw.get_xlim()
    ax_cdw.set_xlim(xlim)
    ax_cdw.plot(xlim, [0,0], "k--", lw=0.5, zorder=-1)
    ax_cdw.text(3,0.5,"Biomass", ha="left")
    for subs, ax in zip(substrates, axes_subs):
        ax.plot(xlim, [0,0], "k--", lw=0.5, zorder=-1)
        ax.text(70,1.4,subs_long_names[subs], ha="right")
    axes[5,0].set_xlabel("hours")
    axes[5,1].set_xlabel("hours")
    
    figname = "depletion_trajectories_mix.svg"
    figname = os.path.join(FIG_DIR, figname)
    plt.savefig(figname)
    print(f"Saved fig {figname}")
    if show:
        plt.show()
            

def testPlotExperiment(experiment, rid):
    t = experiment._tSpan[rid]
    s = experiment._sSpan[rid]
    cdw = experiment._cdwSpan[rid]
    cdwod = experiment._cdwodSpan[rid]
    tda = experiment._tdaSpan[rid]
    
    fac = min([1.0/np.nanmax(sv) for sv in s.values()])

    fig, ax = plt.subplots()
    for sl, sv in s.items():
        plt.plot(t, fac*sv, label=sl, color=config.substrate_colors[sl])
    plt.plot(t, cdw, "x", label="CDW", color=config.substrate_colors["CDW"])
    plt.plot(t, cdwod, label="OD", color=config.substrate_colors["CDW"])
    plt.plot(t, tda, label="TDA", color=config.substrate_colors["TDA"])
    plt.legend()
    plt.title("'%s', run '%s'"%(experiment._experimentID, rid))
    
    tr = experiment._tSpanRates[rid]
    sr = experiment._sRates[rid]
    cdwr = experiment._cdwRates[rid]
    cdwodr = experiment._cdwodRates[rid]
    tdar = experiment._tdaRates[rid]
    
    fig, ax = plt.subplots()
    for sl, sv in sr.items():
        plt.plot(tr, fac*sv, label=sl, color=config.substrate_colors[sl])
    plt.plot(tr, cdwr, "x", label="CDW", color=config.substrate_colors["CDW"])
    plt.plot(tr, cdwodr, label="OD", color=config.substrate_colors["CDW"])
    plt.plot(tr, tdar, label="TDA", color=config.substrate_colors["TDA"])
    plt.legend()
    plt.title("'%s', run '%s', rates"%(experiment._experimentID, rid))
    
    sro = experiment._sRatesOD[rid]
    cdwro = experiment._cdwRatesOD[rid]
    cdwodro = experiment._cdwodRatesOD[rid]
    tdaro = experiment._tdaRatesOD[rid]
    
    fig, ax = plt.subplots()
    for sl, sv in sro.items():
        plt.plot(tr, fac*sv, label=sl, color=config.substrate_colors[sl])
    plt.plot(tr, cdwro, "x", label="CDW", color=config.substrate_colors["CDW"])
    plt.plot(tr, cdwodro, label="OD", color=config.substrate_colors["CDW"])
    plt.plot(tr, tdaro, label="TDA", color=config.substrate_colors["TDA"])
    plt.legend()
    plt.title("'%s', run '%s', rates OD"%(experiment._experimentID, rid))
    
    
def plotCluster(fitter, rid, omitted, separateFigs=True, show=False, colors=None):
    plt.close("all")
    print("Plotting clusters for '%s'"%fitter.getID())
    
    if omitted:
        omit_str = ",".join(omitted)
        omit_str = f"_omitted[{omit_str}]" 
    else:
        omit_str = ""

    experiment = fitter.getExperiment()
    substrates = fitter.getSubstrates()
    clusters = fitter.getClusters()

    print("clusters:", clusters)
    
    if not separateFigs:
        fig, ax = plt.subplots()
        ax2 = plt.twinx(ax)
    
    ix = 0
    for cid, cix in clusters.items():
        if separateFigs:
            fig, ax = plt.subplots()
            ax2 = plt.twinx(ax)
        clusterColor = config.substrate_colors[cid] if colors is None else colors[cid]
        ix += 1
        plotSubstrateCurveData(experiment, rid, [substrates[i] for i in cix], ax, False, "-x", clusterColor=clusterColor)
        
        if separateFigs:
            plotGrowthCurveData(experiment, rid, False, ax2, "OD", "-x")
            plt.title("Substrate cluster '%s' concentration curves (%s, %s)"%(cid, fitter.getID(), rid))
            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(handles=handles+handles2, labels=labels+labels2, loc="upper left", bbox_to_anchor=(1.1, 1.0))
            fig.set_size_inches(plotDimensions["timeline"])
            ax.set_ylabel("concentration $S_i$ [mM/l]")
            ax.set_xlabel("time $t$ [h]")
            ax2.set_ylabel("OD")
            ax.set_xlim([0.0, ax.get_xlim()[1]])
            plt.tight_layout()
            align_yaxis(ax, ax2)
            figname = f"Cluster{omit_str}_%s_%s_%s"%(cid, fitter.getID(), rid)
            figname = os.path.join(FIG_DIR, figname+".svg")
            plt.savefig(figname)
    
    if not separateFigs:
        plotGrowthCurveData(experiment, rid, False, ax2, "OD", "-x")
        plt.title("Substrate cluster concentration curves (%s, %s)"%(fitter.getID(), rid))
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles=handles+handles2, labels=labels+labels2, loc="upper left", bbox_to_anchor=(1.1, 1.0))
        fig.set_size_inches(plotDimensions["clusterTimeline"])
        ax.set_ylabel("concentration $S_i$ [mM/l]")
        ax.set_xlabel("time $t$ [h]")
        ax2.set_ylabel("OD")
        ax.set_xlim([0.0, ax.get_xlim()[1]])
        plt.tight_layout()
        align_yaxis(ax, ax2)
        figname = f"Cluster{omit_str}_%s_%s"%(fitter.getID(), rid)
        figname = os.path.join(FIG_DIR, figname+".svg")
        plt.savefig(figname)
    if show:
        plt.show()
        
def plotGrowthFit(fitter, growthParams, errors, runIDs, plotLabel, logscale=False, show=False):
    plt.close("all")
    print("Plotting growth fit '%s'"%fitter.getID())
    if type(growthParams["V0"]) != dict:
        growthParams["V0"] = {fitter.getID() : dict(zip(fitter.getRunIDs(), growthParams["V0"]))}
    fitter.setGrowthParams(growthParams)
    if fitter.getID() in growthParams["V0"]:
        V0 = growthParams["V0"][fitter.getID()]
    else:
        V0 = growthParams["V0"]
    if not isinstance(V0, dict):
        V0 = dict([(rid, V0[i]) for i,rid in enumerate(fitter.getRunIDs())])
    fitter.setV0(V0)
    
    experiment = fitter.getExperiment()
    includeTDA = fitter._tdaFitting
    substrates = fitter.getSubstrates()
    
    ## Concentration plot (linear scale)
    # Plot data
    fig, ax2 = plt.subplots(layout="constrained")    
    ax = plt.twinx(ax2)
    if logscale:
        ax2.set_yscale("log")

    ax.plot((fitter.getTSpan("F1")[0], fitter.getTSpan("F1")[-1]), [0,0], "k--", lw=0.5)
    for run_nr, rid in enumerate(runIDs):
        plotGrowthCurveData(experiment, rid, includeTDA, ax, biomassIndicator="OD", 
                            colorShade=config.run_colors[rid])
        # Generate fitted solution
        ssr, sst = 0.0, 0.0
        sln, rhs = fitter.generateBiomassSolution(rid)
        ssr, sst = fitter.residualErrorBiomass(rid)
        errors = {"ssr":ssr, "sst":sst}
    
        # Plot simulated solution
        plotGrowthCurveSim(sln, includeTDA, ax, colorShade=config.run_colors[rid], label=rid, plotE=True)
        ax.legend()
        plotGrowthRateSim(sln, rhs, plotTDA=False, ax=ax2, color=config.run_colors[rid])
        ax2.set_ylabel("specific growth rate $[\\dot{V}]$")
        ax2.legend()
        
        title = "Growth curve"
        if errors is not None:
            title += "\n(%s, %s, $R^2=%.4f$)"%(plotLabel, rid, 1 - errors["ssr"]/errors["sst"])
        else:
            title += "\n(%s, %s)"%(plotLabel, rid)
        plt.suptitle(title)
    utils.align_yaxis(ax, ax2)
    fig.set_size_inches(plotDimensions["timeline_growth"])
    figname = os.path.join(FIG_DIR, "growth_fit_%s.svg"%("_".join(plotLabel.split(" "))))
    plt.savefig(figname)
    print("saved fig '%s'"%figname)
    if show:
        plt.show()
    
    
def plotSingleSubstrateUptakeFit(fitter, rid, sid, sol, rhs, title, fn):
    solT = sol.t[:fitter._ixMaxGrowth[rid]]
    tplot = np.linspace(solT[0],solT[-1],101)
    splot = sol.sol(tplot)
    fig, ax = plt.subplots()
    ax.plot((solT[0],solT[-1]), [0, 0], "k:")
    ax.set_xlim((solT[0],solT[-1]))

    ax.plot(fitter.getTSpan(rid), fitter.getS(rid)[sid], ls="--", marker="x", label="%s (exp)"%sid, color=config.substrate_colors[sid])
    # plot inhibiting substrates
    ints = fitter.getInteractions()
    if sid in ints:
        ints_i = ints[sid]
        assert(len(ints_i) == 1) # Currently this works only for one interaction
        ints_i = ints_i[0]
        inh = [fitter.getSubstrates()[six] for six in ints_i["clusterIndices"][0]]
        for inh_id in inh:
            ax.plot(fitter.getTSpan(rid), fitter.getS(rid)[inh_id], ls="--", marker="x", label="%s (exp)"%inh_id, color=config.substrate_colors[inh_id])
    ax.plot(tplot, splot.T, label="%s (sim)"%sid, color=config.substrate_colors[sid])
    ax.legend()
    ax.set_ylabel("Substrate concentration $S$")
    fig.suptitle(title)
    
    ax2 = plt.twinx(ax)
    
    # Make dimmed color for the rates
    basecolor=config.substrate_colors[sid][:3]
    grey = sum(basecolor)/3
    sat, light = 0.2, 3
    dimmed_color = [min([light*(v*sat + grey*(1-sat)), 1.0]) for v in config.substrate_colors[sid][:3]]
    
    iPolVplot = fitter._dynamics._iPolV[rid](tplot)
    dsfull = np.array([rhs(t,s)/v for t, s, v in zip(tplot, splot.T, iPolVplot)]).T
    ax2.plot(tplot, dsfull.T, color=dimmed_color, lw=0.3)
    
    tfit = np.linspace(solT[fitter._ixMinRates[rid]],solT[-1],101)
    sfit = sol.sol(tfit)
    iPolVfit = fitter._dynamics._iPolV[rid](tfit)
    dsfit = np.array([rhs(t,s)/v for t, s, v in zip(tfit, sfit.T, iPolVfit)]).T
    ax2.plot(tfit, dsfit.T, label="(sim)", color=dimmed_color)#, lw=1)
    
    tSpanRates = fitter.getTSpanRates(rid)[fitter._ixMinRates[rid]:fitter._ixMaxGrowth[rid]]
    sRatesODExp = fitter.getSRatesOD(rid)[sid][fitter._ixMinRates[rid]:fitter._ixMaxGrowth[rid]]
    
    ax2.plot(tSpanRates, sRatesODExp, label="(exp)", color=dimmed_color, marker="x", ls=":")
    ax2.set_xlabel("time $t$ [h]")
    ax2.set_ylabel("Specific depletion rate $-\\dot{S}/V$")
    ax2.legend(title="Spec.depl.rate")
    
    utils.align_yaxis(ax, ax2)
    
    fig.set_size_inches(plotDimensions["timeline"])
    
    if fn is not None:
        fig.savefig(fn)
        print("Saved figure '%s'"%fn)
    

def plotFullFit(results_fn, multiSubstrateFitter, full_fit_type, legend=False):
    # full_fit_types must match full_fit_types in main_mix11_analysis.py
    full_fit_types = [
        "no refit with interactions",
        "refit without interactions",
        "refit with interactions"
    ]

    results = utils.loadState(results_fn)
    if full_fit_type not in results:
        raise Exception(f"Cannot plot ''{full_fit_type}' fit results, run fitFullModel() first.")
    
    # Ensure that clusters are computed
    runIDs = multiSubstrateFitter.getRunIDs()

    # Fitter was prepared already    
    sln, rhs = {}, {}
    ssr, sst = {}, {}
    for rid in runIDs:
        sln[rid], rhs[rid] = multiSubstrateFitter.generateFullSolution(rid, False)
        ssr[rid], sst[rid] = fitting.fullResidualError(multiSubstrateFitter, rid)
    
    experiment = multiSubstrateFitter.getExperiment()
    substrates = multiSubstrateFitter.getSubstrates()
    ix0 = 2
    substrateIndices=list(range(ix0, ix0+len(substrates)))
    
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(8,10), layout="constrained")
    gs = GridSpec(8, 2, figure=fig)
    
    ylim_cdw, ylim_cdw_rates = [0,0], [0,0]
    ylim_subs, ylim_subs_rates = [0,0], [0,0]
    axs = dict(
        cdw = [None]*4,
        cdw_rates = [None]*4,
        subs = [None]*4,
        subs_rates = [None]*4,
        )
    for run, rid in enumerate(runIDs):
        axs["cdw"][run] = fig.add_subplot(gs[2*run:2*(run+1),0])
        ax_cdw = axs["cdw"][run] 
        plotGrowthCurveData(experiment=experiment, rid=rid, ax=ax_cdw, plotTDA=False,
                            biomassIndicator="OD", colorShade=config.substrate_colors["OD "])
        plotGrowthCurveSim(sln[rid], plotTDA=False, ax=ax_cdw, logscale=False, plotE=True, 
                           colorShade="gray", label=f"run {run +1}")
        ax_cdw.set_xlabel(None)
        ax_cdw.text(2.0, 0.5, f"replicate {run +1}")
        ax_cdw_rates = ax_cdw
        axs["cdw_rates"][run] = ax_cdw_rates 
        plotGrowthRateData(experiment, rid, False, ax_cdw_rates, 
                           biomassIndicator="OD", ls="x", color="k")
        plotGrowthRateSim(sln[rid], rhs[rid], False, ax_cdw_rates, True, ls="--")
        ax_cdw.set_ylabel(None)
        
        axs["subs"][run] = fig.add_subplot(gs[2*run,1], in_layout=True)
        ax_subs = axs["subs"][run]
        ax_subs.set_xlabel(None)
        ax_subs.set_ylabel("concentration\n$S$ [mM]")
        plotSubstrateCurveData(experiment, rid, substrates, 
                               aggregateSubstrates=False,
                               ax=ax_subs, ls="x")
        plotSubstrateCurveSim(sln[rid], substrateIndices=substrateIndices, 
                              aggregateSubstrates=False,
                              substrates=substrates, ax=ax_subs)
        ax_subs_rates = fig.add_subplot(gs[2*run+1,1], in_layout=True)
        ax_subs_rates.set_ylabel("depletion rate\n$-dS/dt$ [mM/h]")
        axs["subs_rates"][run] = ax_subs_rates
        plotSubstrateRateData(experiment, rid, substrates, 
                              ax_subs_rates, False, ls="x", invert=True)
        plotSubstrateRateSim(sln[rid], rhs[rid], substrateIndices, 
                             substrates, ax_subs_rates, False, invert=True)
        ax_subs_rates.set_xlabel(None)
        
        ylim_cdw = [min(ylim_cdw[0], ax_cdw.get_ylim()[0]), 
                    max(ylim_cdw[1], ax_cdw.get_ylim()[1])]        
        ylim_cdw_rates = [min(ylim_cdw_rates[0], ax_cdw_rates.get_ylim()[0]), 
                          max(ylim_cdw_rates[1], ax_cdw_rates.get_ylim()[1])]
        ylim_subs = [min(ylim_subs[0], ax_subs.get_ylim()[0]), 
                    max(ylim_subs[1], ax_subs.get_ylim()[1])]        
        ylim_subs_rates = [min(ylim_subs_rates[0], ax_subs_rates.get_ylim()[0]), 
                          max(ylim_subs_rates[1], ax_subs_rates.get_ylim()[1])] 
    
    xlim=ax_cdw.get_xlim()
    
    ylim_cdw = [min(ylim_cdw[0], ylim_cdw_rates[0]),
                max(ylim_cdw[1], ylim_cdw_rates[1])]
    
    ylim_subs = [-0.1, 1.8]
    ylim_subs_rates = [-0.2, 2.0]
    xlim = [0, 75]
    for run, rid in enumerate(runIDs):
        axs["cdw"][run].set_ylim(ylim_cdw)
        axs["cdw"][run].set_xlim(xlim)
        axs["cdw_rates"][run].set_ylim(ylim_cdw)
        axs["cdw_rates"][run].set_xlim(xlim)
        axs["subs"][run].set_ylim(ylim_subs)
        axs["subs"][run].set_xlim(xlim)
        axs["subs_rates"][run].set_ylim(ylim_subs_rates)
        axs["subs_rates"][run].set_xlim(xlim)
        axs["cdw"][run].plot(xlim, [0,0], "k--", lw=0.5, zorder=-2)
        axs["subs"][run].plot(xlim, [0,0], "k--", lw=0.5, zorder=-2)
        axs["subs_rates"][run].plot(xlim, [0,0], "k--", lw=0.5, zorder=-2)
        axs["subs"][run].set_xticklabels([])
        if run<3:
            axs["subs_rates"][run].set_xticklabels([])
            axs["cdw"][run].set_xticklabels([])
        else:
            axs["subs_rates"][run].set_xlabel("hours")
            axs["cdw"][run].set_xlabel("hours")
        
    if legend:
        axs["subs"][run].legend(ncols=4)
        axs["subs_rates"][run].legend(ncols=4)
        axs["cdw"][run].legend()
        axs["cdw_rates"][run].legend()
    
    if legend:
        figname = "full_fit_mix11_with_interaction_with_legend.svg"
    else:
        figname = "full_fit_mix11_with_interaction.svg"
    figname = FIG_DIR / figname
    fig.savefig(figname)
    print(f"Saved fig '{figname}'")
    
    
def plotSubstrateCurveData(experiment, rid, substrates, ax, aggregateSubstrates, ls="x", 
                           logscale=False, clusterColor=None, labels=True, plot_obs_range=False):    
    t = experiment._tSpan[rid]
    s = experiment._sSpan[rid]

    if not logscale:
        if aggregateSubstrates:
            aggregatedS = np.zeros(len(t))
            for l in substrates:
                aggregatedS += s[l]
            color = "blue" if clusterColor is None else clusterColor
            ax.plot(t, aggregatedS, ls, label="total substrate [Cmol]", color=color)
        else:
            for l in substrates:
                hash_color = config.hash_color(l+"asfda")
                color = config.substrate_colors.get(l, hash_color) if clusterColor is None else clusterColor
                
                lab = l if labels else None
                if ls:
                    ax.plot(t, s[l], ls, label=lab, color=color)
                if plot_obs_range:
                    exclude_runs = ["F4"] if l=="His" else []
                    all_s = defaultdict(list)
                    for rid in experiment._runIDs:  
                        if rid in exclude_runs:
                            continue
                        tr = experiment._tSpan[rid]
                        sr = experiment._sSpan[rid][l]
                        for ti, si in zip(tr,sr):
                            all_s[ti].append(si)
                    assert(np.all([len(si)==len(all_s[0.0]) for si in all_s.values()]))
                    ts = sorted(all_s)
                    all_s = np.array([all_s[ti] for ti in ts])
                    means, stds = np.mean(all_s, axis=1), np.std(all_s, axis=1)
                    ax.plot(ts, means, ls="--", color=color, lw=0.8, alpha=0.5, label = f"Mean '{l}'")
                    if plot_obs_range=="std":
                        conf_up, conf_down = means + stds, means - stds
                        ax.fill_between(ts, conf_down, conf_up, color=color, alpha=0.3, zorder=-1)
                    else:
                        maxima, minima = np.max(all_s, axis=1), np.min(all_s, axis=1)
                        ax.fill_between(ts, minima, maxima, color=color, alpha=0.3, zorder=-1, label = f"Range '{l}'")



    else:
        # We plot an expression, which gives an approximation for mu*t 
        # according to the following reasoning:
        #     (1)  V  = V0*exp(mu*t) => mu*t = ln(V/V0)
        #     (2)  S' = -nu*V = -nu*V0*exp(mu*t)
        #  =>  S = S0*exp(-nu*V0*(exp(mu*t)-1)/mu)
        #  =>  ln(S0/S) = nu*V0*(exp(mu*t)-1)/mu
        #  =>  ln(S0/S)*mu/(nu*V0) + 1 = exp(mu*t)
        #  =>  mu*t = ln(ln(S0/S)*mu/(nu*V0) + 1)   (3)  
        # To determine mu/nu (the effective yield in case of one substrate, btw), 
        # we use (1) with (3):
        #  =>  V/V0 = ln(S0/S)*mu/(nu*V0) + 1
        #  =>  V = V0 + ln(S0/S)*mu/nu
        #  =>  mu/nu = (V-V0)/ln(S0/S) (4)
        #  From (4) we estimate y:=mu/nu to use in (3), where we only consider 
        # data where tmax>t>tmin (from experiment)
        minix, maxix = fitting.getIxMinAndIxMax(experiment, fitting.MIN_CDWOD_RATES)
        V = experiment._cdwodSpan[rid][minix[rid]:maxix[rid]]
        t = t[minix[rid]:maxix[rid]]
        
        if aggregateSubstrates:
            S = np.zeros(len(t))
            for l in substrates:
                S += s[l][minix[rid]:maxix[rid]]
            if np.argmax(S<=0) > 0: maxixAggr = min(maxix[rid]-minix[rid], np.argmax(S<=0))
            else: maxixAggr = maxix[rid]-minix[rid]
            y = np.mean((V[1:maxixAggr]-V[0])/np.log(S[0]/S[1:maxixAggr]))
            mut = np.log(1 + np.log(S[0]/S[:maxixAggr])*y/V[0])
            print("substrates:", substrates)
            print("S:", S[:maxixAggr])
            print("np.log(S[0]/S[1:maxixAggr]) =", np.log(S[0]/S[1:maxixAggr]))
            print("mu*t:", mut)
            color = "blue" if clusterColor is None else clusterColor
            ax.plot(t[:maxixAggr], mut, ls, label="$\\mu*t$ (aggregated substrates)", color=color)
            
        else:
            for l in substrates:
                S = s[l][minix[rid]:maxix[rid]]
                if np.argmax(S<=0) > 0: maxixAggr = min(maxix[rid]-minix[rid], np.argmax(S<=0))
                else: maxixAggr = maxix[rid]-minix[rid]
                y = np.mean((V[1:maxixAggr]-V[0])/np.log(S[0]/S[1:maxixAggr]))
                mut = np.log(1 + np.log(S[0]/S[:maxixAggr])*y/V[0])
                color = config.substrate_colors[l] if clusterColor is None else clusterColor
                ax.plot(t[:maxixAggr], mut, ls, label="$\\mu*t$ (%s)"%l, color=color)



def plotSubstrateRateData(experiment, rid, substrates, ax, aggregateSubstrates, ls="x", 
                          labels=True, invert=True, plot_obs_range=False, t_offset=0.0):
    tr = experiment._tSpanRates[rid]
    tix = [t >= t_offset for t in tr]
    tr = np.array(tr)[tix] 
    sr = experiment._sRatesOD[rid]
    sr = {k: np.array(v)[tix] for k, v in  sr.items()}
    sign = -1 if invert else 1
    
    if aggregateSubstrates:
        aggregatedSR = np.zeros(len(tr))
        for l in substrates:
            aggregatedSR += sr[l]
        if ls:
            ax.plot(tr, sign*aggregatedSR, ls, label="total depletion rate [Cmol]", color="blue")
    else:
        for l in substrates:
            hash_color = config.hash_color(l+"asfda")
            lab = l if labels else None
            color = config.substrate_colors.get(l, hash_color)
            if ls:
                ax.plot(tr, sign*sr[l], ls, label=lab, color=color)
            if plot_obs_range:
                exclude_runs = ["F4"] if l=="His" else []
                all_sr = defaultdict(list)
                for rid in experiment._runIDs:
                    if rid in exclude_runs:
                        continue
                    tr = experiment._tSpanRates[rid]
                    tix = [t >= t_offset for t in tr]
                    tr = np.array(tr)[tix] 
                    sr = experiment._sRatesOD[rid][l]
                    sr = np.array(sr)[tix]
                    for ti, si in zip(tr,sr):
                        all_sr[ti].append(sign*si)
                ts = [t for t in sorted(all_sr) if t >= t_offset]
                assert(np.all([len(si)==len(all_sr[ts[0]]) for si in all_sr.values()]))
                all_sr = np.array([all_sr[ti] for ti in ts])
                means, stds = np.mean(all_sr, axis=1), np.std(all_sr, axis=1)
                ax.plot(ts, means, ls="--", color=color, lw=0.8, alpha=0.5, label = f"Mean '{l}'")
                if plot_obs_range=="std":
                    conf_up, conf_down = means + stds, means - stds
                    ax.fill_between(ts, conf_down, conf_up, color=color, alpha=0.3, zorder=-1)
                else:
                    maxima, minima = np.max(all_sr, axis=1), np.min(all_sr, axis=1)
                    ax.fill_between(ts, minima, maxima, color=color, alpha=0.3, zorder=-1, label = f"Range '{l}'")
    

def plotGrowthCurveData(experiment, rid, plotTDA, ax, biomassIndicator="CDW,OD", ls="x", 
                        logscale=False, colorShade=None, labels=True, plot_obs_range=False): 
    plotCDW = "CDW" in biomassIndicator    
    plotOD  = "OD"  in biomassIndicator
        
    t = experiment._tSpan[rid]
    cdw = experiment._cdwSpan[rid]
    if rid in experiment._cdwodSpan:
        cdwod = experiment._cdwodSpan[rid]
    minix, _ = fitting.getIxMinAndIxMax(experiment, fitting.MIN_CDWOD_RATES)
    minix = minix[rid]
    if plotTDA:
        tda = experiment._tdaSpan[rid]

    if ls == "-x":
        ls="-"
        marker="x"
    else:
        marker=None
    
    if plotOD:
        color = config.substrate_colors["OD "] if colorShade is None else colorShade
        label = rid if labels else None
        if logscale:
            ax.plot(t, np.log(cdwod/cdwod[minix]), ls=ls, label=label, color=color, marker=marker)
        else:
            ax.plot(t, cdwod, ls=ls, label=label, color=color, marker=marker)
    if plotCDW:
        color = config.substrate_colors["CDW"] if colorShade is None else colorShade
        label = rid if labels else None
        if logscale:
            ax.plot(t, np.log(cdw/cdw[minix]), ls=ls, label=label, color=color, marker=marker)
        else:
            ax.plot(t, cdw, ls=ls, label=label, color=color, marker=marker)
    if plotTDA:
        if logscale:
            ax.plot(t, np.log(tda/tda[minix]), ls=ls, label="TDA"  if labels else None, color=config.substrate_colors["TDA"], marker=marker)
        else:
            ax.plot(t, tda, ls=ls, label="TDA", color=config.substrate_colors["TDA"], marker=marker)

    if plot_obs_range:
        color = config.substrate_colors["CDW"] if colorShade is None else colorShade
        all_cdw = defaultdict(list)
        for rid in experiment._runIDs:
            tr = experiment._tSpan[rid]
            cdw = experiment._cdwodSpan[rid]
            for ti, ci in zip(tr,cdw):
                all_cdw[ti].append(ci)
        assert(np.all([len(ci)==len(all_cdw[0.0]) for ci in all_cdw.values()]))
        ts = sorted(all_cdw)
        all_cdw = np.array([all_cdw[ti] for ti in ts])
        means, stds = np.mean(all_cdw, axis=1), np.std(all_cdw, axis=1)
        ax.plot(ts, means, ls="--", color=color, lw=0.8, alpha=0.5, label = f"Mean CDW [g/L]")
        if plot_obs_range=="std":
            conf_up, conf_down = means + stds, means - stds
            ax.fill_between(ts, conf_down, conf_up, color=color, alpha=0.3, zorder=-1)
        else:
            maxima, minima = np.max(all_cdw, axis=1), np.min(all_cdw, axis=1)
            ax.fill_between(ts, minima, maxima, color=color, alpha=0.3, zorder=-1, label = f"Range CDW [g/L]")

    
def plotGrowthRateData(experiment, rid, plotTDA, ax, 
                       biomassIndicator="CDW,OD", ls="x", 
                       labels=True, color=None, plot_obs_range=False):     
    plotCDW = "CDW" in biomassIndicator    
    plotOD  = "OD"  in biomassIndicator
    
    tr = experiment._tSpanRates[rid]
    cdwr = experiment._cdwRates[rid]
    if rid in experiment._cdwodRatesOD:
        cdwodr = experiment._cdwodRatesOD[rid]
    if plotTDA:
        tdar = experiment._tdaRates[rid]
    
    if ls == "-x":
        ls="-"
        marker="x"
    else:
        marker=None

    if plotOD:
        clr = config.substrate_colors["OD "] if color is None else color
        plt.plot(tr, cdwodr, ls=ls, label="OD" if labels else None, color=clr, marker=marker)
    if plotCDW:
        clr = config.substrate_colors["CDW"] if color is None else color
        plt.plot(tr, cdwr, ls=ls, label="CDW" if labels else None, color=clr, marker=marker)
    if plotTDA:
        clr = config.substrate_colors["TDA"] if color is None else color
        plt.plot(tr, tdar, ls=ls, label="TDA" if labels else None, color=clr, marker=marker)

    if plot_obs_range:
        color = config.substrate_colors["CDW"]
        all_cdw = defaultdict(list)
        for rid in experiment._runIDs:
            tr = experiment._tSpanRates[rid]
            cdw = experiment._cdwodRatesOD[rid]
            for ti, ci in zip(tr,cdw):
                all_cdw[ti].append(ci)
        assert(np.all([len(ci)==len(all_cdw[tr[0]]) for ci in all_cdw.values()]))
        ts = sorted(all_cdw)
        all_cdw = np.array([all_cdw[ti] for ti in ts])
        means, stds = np.mean(all_cdw, axis=1), np.std(all_cdw, axis=1)
        ax.plot(ts, means, ls="--", color=color, lw=0.8, alpha=0.5, label = f"Mean growth rate [g/L/h]")
        if plot_obs_range=="std":
            conf_up, conf_down = means + stds, means - stds
            ax.fill_between(ts, conf_down, conf_up, color=color, alpha=0.3, zorder=-1)
        else:
            maxima, minima = np.max(all_cdw, axis=1), np.min(all_cdw, axis=1)
            ax.fill_between(ts, minima, maxima, color=color, alpha=0.3, zorder=-1, label = f"Range growth rate [g/L/h]")


def plotGrowthCurveSim(sln, plotTDA, ax, logscale=False, plotE=False, colorShade=None, label=None):
    Nt = 100
    t = np.linspace(sln.t[0], sln.t[-1], Nt)
    solSpan = sln.sol(t)
    cdw = solSpan[0]
    e = solSpan[1]/solSpan[0]
    if plotTDA:
        tda = solSpan[2]
    
    minix = np.argmax(solSpan[0] >= fitting.MIN_CDWOD_RATES)
    
    eFac = np.max(cdw)/np.max(e)
    
    if logscale:
        label="CDW" if label is None else label
        color = config.substrate_colors["CDW"] if colorShade is None else colorShade
        ax.plot(t, np.log(cdw/cdw[minix]), label=label, color=color)
        ax.set_ylabel("CDW (log)")
        if plotTDA:
            ax.plot(t, np.log(tda/tda[minix]), label="TDA", color=config.substrate_colors["TDA"])
    else:
        label="CDW" if label is None else label
        color = config.substrate_colors["CDW"] if colorShade is None else colorShade
        ax.plot(t, cdw, label=label, color=color)
        ax.set_ylabel("CDW")
        ax.set_xlabel("time $t$")
        if plotE:
            ax.plot(t, eFac*e, label="[E]", color=config.substrate_colors["E"], ls="--")
        if plotTDA:
            tdaFac = np.max(cdw)/np.max(tda)
            ax.plot(t, tdaFac*tda, label="TDA", color=config.substrate_colors["TDA"])

def plotSubstrateCurveSim(sln, substrateIndices, substrates, ax, 
                          aggregateSubstrates, logscale=False, 
                          growthParams=None, color=None, label=True):
    if len(substrates) == 0:
        return
    
    Nt = 100
    t = np.linspace(sln.t[0], sln.t[-1], Nt)
    solSpan = sln.sol(t)
    
    if not logscale:
        if aggregateSubstrates:
            aggregatedS = np.zeros(len(t))
            for i in substrateIndices:
                aggregatedS += solSpan[i]
            ax.plot(t, aggregatedS, label="total substrate sim [Cmol]", color="blue")
        else:
            for i, l in zip(substrateIndices, substrates):
                col = config.substrate_colors[l] if l in config.substrate_colors else config.hash_color(l+"asfda")
                lab = f"Fit '{l}'" if label else None
                ax.plot(t, solSpan[i], label=lab, color=col)
    elif growthParams is not None:
        # We plot an expression, which gives an approximation for mu*t
        # @see: plotSubstrateCurveData()
        minix = np.argmax(solSpan[0] >= fitting.MIN_CDWOD_RATES)
        maxix = np.argmax(solSpan[0]) + 1
        t = t[minix:maxix]
        V = solSpan[0][minix:maxix]
            
        if aggregateSubstrates:
            S = np.zeros(len(t))
            for i in substrateIndices:
                S += solSpan[i][minix:maxix]
            sixmax = np.argmax(S < 0) if (np.argmax(S < 0) > 0) else maxix-minix
            y = np.mean((V[1:sixmax]-V[0])/np.log(S[0]/S[1:sixmax]))
            mut = np.log(1 + np.log(S[0]/S[1:sixmax])*y/V[0])
            ax.plot(t[1:sixmax], mut, label="$\\mu*t$ (aggregated substrates, sim)", color="blue")
            
        else:
            for i,l in zip(substrateIndices, substrates):
                S = solSpan[i][minix:maxix]
                sixmax = np.argmax(S < 0) if (np.argmax(S < 0) > 0) else maxix-minix
                y = np.mean((V[1:sixmax]-V[0])/np.log(S[0]/S[1:sixmax]))
                mut = np.log(1 + np.log(S[0]/S[1:sixmax])*y/V[0])
                col = config.substrate_colors[l] if l in config.substrate_colors else config.hash_color(l+"asfda")
                ax.plot(t[1:sixmax], mut, label="$\\mu*t$ (%s, sim)"%l, color=col)
            
    
def plotSubstrateRateSim(sln, rhs, substrateIndices, substrates, 
                         ax, aggregateSubstrates, odSpanInterpolation = None,
                         invert=True, label=True):
    Nt = 100
    t = np.linspace(sln.t[0], sln.t[-1], Nt)
    solSpan = sln.sol(t)
    sign = -1 if invert else 1
    
    # check for substrate only simulation
    if odSpanInterpolation is not None:
        # The interpolated OD of the experiment is used to calculate 
        # the specific rates, if the solution does not contain information on the biomass.
        odSpan = odSpanInterpolation(t)
        rateSpan = np.array(list(map(rhs, t, solSpan.T))).T/odSpan
    else:
        rateSpan = np.array(list(map(rhs, t, solSpan.T))).T/solSpan[0]
    
    if aggregateSubstrates:
        aggregatedRates = np.zeros(len(t))
        for i in substrateIndices:
            aggregatedRates += rateSpan[i]
        ax.plot(t, sign*aggregatedRates, label="total depletion rate sim [Cmol]", color="blue")
    else:
        for i, l in zip(substrateIndices, substrates):
            col = config.substrate_colors[l] if l in config.substrate_colors else config.hash_color(l+"asfda")
            lab = f"Fit '{l}'" if label else None
            ax.plot(t, sign*rateSpan[i], label=lab, color=col)
    
    
def plotGrowthRateSim(sln, rhs, plotTDA, ax, plotE=False, ls="-", color=None):
    Nt = 100
    t = np.linspace(sln.t[0], sln.t[-1], Nt)
    solSpan = sln.sol(t)
    rateSpan = np.array(list(map(rhs, t, solSpan.T))).T/solSpan[0]
    fac=1.0
    color = config.substrate_colors["CDW"] if color is None else color
    ax.plot(t, fac*rateSpan[0], label="specific growth rate", 
            ls=ls, color=color)
    
def plotParameterComparison(paramsMix, paramsSingle, comparisonID = ""):
    plt.close("all")
    
    df = {"Substrate":[], "Culture":[], "$K$":[], "$\\mu$":[]}
    df_mu = {"Substrate":[], "$\\mu_1/\\mu_2$":[]}
    df_growth = {"$y_V$":[], "m":[], "$r_E$":[], "Culture":[]}
    
    singleSubstrates = [k for k in paramsSingle.keys() if len(k)==3]
    for s in singleSubstrates:
        df["Substrate"].extend((s, s))
        df["Culture"].extend(("pure", "mix"))
        df["$K$"].append(paramsSingle[s][s]["K"])
        df["$\\mu$"].append(paramsSingle[s][s]["mu"])
        df["$K$"].append(paramsMix[s]["K"])
        df["$\\mu$"].append(paramsMix[s]["mu"])
        
        df_mu["Substrate"].append(s)
        df_mu["$\\mu_1/\\mu_2$"].append(paramsMix[s]["mu"]/paramsSingle[s][s]["mu"])

    
    df_growth["m"].append(1.0)
    df_growth["$y_V$"].append(1.0)
    df_growth["$r_E$"].append(1.0)
    df_growth["Culture"].append("pure")
    df_growth["m"].append(paramsMix["m"]/paramsSingle["m"])
    df_growth["$y_V$"].append(paramsMix["yV"]/paramsSingle["yV"])
    df_growth["$r_E$"].append(paramsMix["rE"]/paramsSingle["rE"])
    df_growth["Culture"].append("mix")
    
    
    df = pd.DataFrame(df)
    df_mu = pd.DataFrame(df_mu)
    df_growth = pd.DataFrame(df_growth)
    df_growth = df_growth.melt(id_vars=["Culture"], value_vars=["m", "$y_V$", "$r_E$"], value_name="Ratio mix:pure")
        
    # Plot mu comparison
    fig, ax = plt.subplots()
    fig.set_size_inches(plotDimensions["table"])
    sns.barplot("Substrate", "$\\mu$", "Culture", df, ax=ax, figure=fig)
    title = "Comparison $\\mu$"
    if comparisonID:
        title += " " + comparisonID
    plt.title(title)
    plt.tight_layout()
    
    figname = "Comparison_uptake_fits_mu_%s"%comparisonID
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)
    
    
    # Plot K comparison
    fig, ax = plt.subplots()
    fig.set_size_inches(plotDimensions["table"])
    sns.barplot("Substrate", "$K$", "Culture", df, ax=ax, figure=fig)
    title = "Comparison $K$"
    if comparisonID:
        title += " " + comparisonID
    plt.title(title)
    plt.tight_layout()
    
    figname = "Comparison_uptake_fits_K_%s"%comparisonID
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)
    
    
    # Plot m comparison
    fig, ax = plt.subplots()
    fig.set_size_inches(plotDimensions["table"])
    sns.barplot("Substrate", "$K$", "Culture", df, ax=ax, figure=fig)
    title = "Comparison $K$"
    if comparisonID:
        title += " " + comparisonID
    plt.title(title)
    plt.tight_layout()
    
    figname = "Comparison_uptake_fits_K_%s"%comparisonID
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)
    
    
    # Plot mu change ratios
    fig, ax = plt.subplots()
    fig.set_size_inches(plotDimensions["table"])
    colors = [config.substrate_colors[s] for s in df_mu["Substrate"]]
    sns.barplot("Substrate", "$\\mu_1/\\mu_2$", data=df_mu, palette=colors, ax=ax, figure=fig)
    title = "Change ratios $\\mu_{single} -> \\mu_{mix}$"
    if comparisonID:
        title += " " + comparisonID
    plt.title(title)
    plt.tight_layout()
    
    figname = "Comparison_uptake_changeratio_mu_%s"%comparisonID
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)
    
    
    # Plot growth parameter comparison
    fig, ax = plt.subplots()
    fig.set_size_inches(plotDimensions["table"])
    sns.barplot("variable", "Ratio mix:pure", "Culture", data=df_growth, ax=ax, figure=fig)
    title = "Comparison growth parameters"
    if comparisonID:
        title += " " + comparisonID
    plt.title(title)
    plt.tight_layout()
    
    figname = "Comparison_growth_params_%s"%comparisonID
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)
    
    
def labelAndAddErrorbars(ax, df_best, plt_var, err_var, substrate=None):
    # Two plot versions:
    # (1) if substrate == None, this is applied to a barplot with substrates on the x-axis, best fits on y-axis
    # (2) if substrate != None, clusters are on the x-axis and corresponding scores on y-axis
    cluster_xaxis = substrate != None

    # label each bar in barplot
    haveOnlyInts = 1 - int(any(df_best["nrInteractions"]==0))
    xticks = ax.get_xticklabels()
    xlabs = [l.get_text() for l in xticks]
    xpos = np.array([l.get_position()[0] for l in xticks])
    for p in list(ax.patches):
        width = p.get_width()
        if width == 0.0:
            # This is no bar
            continue
        x = p.get_x()
        six = np.argmin(np.abs(x-xpos))
        dx = x-xpos[six]
        sid = xlabs[six]
        if dx <= -0.2:
            nInt = 0 + haveOnlyInts
        elif dx <= 0:
            nInt = 1 + haveOnlyInts
        else:
            nInt = 2
        
        if cluster_xaxis:
            df = df_best.loc[df_best["interactions"]==(sid[1:-1],)]
            df = df.loc[df_best["substrate"]==substrate]
        else:
            df = df_best.loc[df_best["substrate"]==sid]
            df = df.loc[df_best["nrInteractions"]==nInt]
        interaction = np.array(df["interactions"])[0]
        error = np.array(df[err_var])[0]
        full_fit_value = np.array(df[plt_var])[0]
        height = p.get_height()
        
        # Add text
        if interaction is not None and not cluster_xaxis:
            interactionstr = ", ".join(interaction)
            # get the height of each bar
            # adding text to each bar
            ax.text(x = p.get_x()+(width/2), # x-coordinate position of data label, padded to be in the middle of the bar
            y = 0.05, # y-coordinate position of data label
            s = str(interactionstr), # data label, formatted to ignore decimals
            ha = "center",
            rotation=90) # sets horizontal alignment (ha) to center
        
        # Add error bars
        ax.errorbar(x+(width/2), height, yerr=error, 
            fmt='none', color='black', capsize=5)


def plotFitImprovement(df_best, df_all, show = False):
    # Plot R2 comparison
    fig, ax = plt.subplots()
    fig.set_size_inches(plotDimensions["table"])
    colors = [to_rgb("xkcd:grey"), to_rgb("xkcd:green"), to_rgb("xkcd:orange")]
    pal = [colors[nInt] for nInt in df_best["nrInteractions"]]
    sns.barplot(x="substrate", y="mean(R2)", hue="nrInteractions", 
                data=df_best, ax=ax, figure=fig, palette=pal)
    # Add error bars and inhibitor labels
    labelAndAddErrorbars(ax, df_best, plt_var="R2", err_var="std(R2)")
    plt.legend(loc="lower right", title="nrInteractions")
    plt.ylabel("$R^2$")
    # plt.ylim((0,1))
    plt.ylim((0,ax.get_ylim()[1]))
    plt.tight_layout()
    
    figname = "uptake_fits_R2"
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)
    
    # Plot error reduction
    df_best = df_best[df_best["nrInteractions"] > 0]
    fig, ax = plt.subplots()
    fig.set_size_inches(plotDimensions["table"])
    pal = [colors[nInt] for nInt in df_best["nrInteractions"]]
    sns.barplot(x="substrate", y="mean(q)", hue="nrInteractions",
                data=df_best, ax=ax, figure=fig, palette=pal)
    labelAndAddErrorbars(ax, df_best, plt_var="rel. error reduction", err_var="std(q)")
    plt.ylabel("rel. error reduction $q$")
    plt.legend(loc="lower right", title="nrInteractions")
    # plt.ylim((0,1))
    plt.ylim((0,ax.get_ylim()[1]))
    plt.tight_layout()

    figname = "uptake_fits_error_reduction"
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)

    # Plot rel fit improvement
    fig, ax = plt.subplots()
    fig.set_size_inches(plotDimensions["table"])
    pal = [colors[nInt] for nInt in df_best["nrInteractions"]]
    sns.barplot(x="substrate", y="mean(p)", hue="nrInteractions", 
                data=df_best, ax=ax, figure=fig, palette=pal)
    labelAndAddErrorbars(ax, df_best, plt_var="rel. fit improvement", err_var="std(p)")
    plt.ylabel("rel. fit improvement $p$")
    plt.legend(loc="lower right", title="nrInteractions")
    # plt.ylim((0,1))
    plt.ylim((0,ax.get_ylim()[1]))
    plt.tight_layout()
    
    figname = "uptake_fits_relative_fit_improvement"
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)

    # Plot fit improvement
    fig, ax = plt.subplots()
    fig.set_size_inches(plotDimensions["table"])
    pal = [colors[nInt] for nInt in df_best["nrInteractions"]]
    sns.barplot(x="substrate", y="dR2", hue="nrInteractions", data=df_best, ax=ax, figure=fig, palette=pal)
    labelAndAddErrorbars(ax, df_best, plt_var="dR2", err_var="std(dR2)")
    plt.ylabel("abs. fit improvement $\\Delta R^2$")
    plt.legend(loc="lower right", title="nrInteractions")
    # plt.ylim((0,1))
    plt.ylim((0,ax.get_ylim()[1]))
    plt.tight_layout()
    
    figname = "uptake_fits_absolute_fit_improvement"
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)

    # Plot dR2 for glucose
    fig, ax = plt.subplots()
    fig.set_size_inches(plotDimensions["table_small"])
    data = df_all.loc[np.logical_and(
        df_all["nrInteractions"]==1, df_all["substrate"]=="Glc"), :]
    data["inhibiting cluster"] = [f"[{c[0]}]" for c in data["interactions"]]
    sns.barplot(x="inhibiting cluster", y="dR2", hue="inhibiting cluster", data=data, ax=ax, figure=fig, 
                palette={c: config.substrate_colors[c[1:-1]] for c in data["inhibiting cluster"]})
    labelAndAddErrorbars(ax, data, plt_var="dR2", err_var="std(dR2)", substrate="Glc")
    # Report std for inhibition coeffcient
    df_glc = data.loc[data["substrate"]=="Glc"]
    df_glc = df_glc[["substrate", "interactions", "R2", "dR2", "a", "mean(a)", "std(a)", "min(a)", "max(a)"]]
    pp("Stats for glucose:")
    pp(df_glc)

    plt.ylabel("abs. fit improvement $\\Delta R^2$")
    plt.legend(loc="lower right", title="inhibitor")
    # plt.ylim((0,0.2))
    plt.ylim((0,ax.get_ylim()[1]))
    plt.tight_layout()
    
    figname = "uptake_fits_absolute_fit_improvement_glucose"
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)

    if show:
        plt.show()
    plt.close("all")
    

def plotR2Overview(fitter, results, onlyBestInteraction=True, substrate="all", nInt=1, show=False):
    # Prepare plotting data
    data = {"Substrate":[], "R2":[], "Rel. model improvement":[], "Abs. error reduction":[], "Model":[]}
    R2NoInteraction = {}
    for s, d in results["uptake"]["errors"]["mix11"].items():
        r2 = 1 - d["ssr"]/d["sst"]
        print("%s (no interactions): R2 = %s"%(s, r2))
        data["Substrate"].append(s)
        data["R2"].append(r2)
        data["Model"].append("No interaction")
        data["Rel. model improvement"].append(np.nan)
        data["Abs. error reduction"].append(np.nan)
        R2NoInteraction[s] = r2
    for s, d in results["interactions"]["errors"].items():
        print("\n%s"%s)
        r2NoInt = R2NoInteraction[s]
        p = results["interactions"]["params"][s]
        bestInteractionModel = {"cluster":None, "R2":0.0}
        for cid, e in d.items():
            print("cluster id:",cid)
            print("params:", p[cid])
            six = fitter.getSubstrateIx(s)
            print("six:", six)
            self_inhibition = np.any([six == cx for cx in p[cid]["clusterIndices"]])
            if self_inhibition:
                # Exclude self inhibition 
                continue
            r2 = max((1 - e["ssr"]/e["sst"], r2NoInt))
            improvement = r2 - r2NoInt 
            errorRedux = (r2 - r2NoInt)/(1-r2NoInt)
            print("%s -> %s: R2 = %s"%(cid, s, r2))
            if not onlyBestInteraction:
                data["Substrate"].append(s)
                data["R2"].append(r2)
                data["Model"].append("cluster '%s'"%str(cid))
                data["Rel. model improvement"].append(errorRedux)
                data["Abs. error reduction"].append(improvement)
            if r2 > bestInteractionModel["R2"]:
                bestInteractionModel["R2"] = r2
                bestInteractionModel["cluster"] = str(cid)
        
        print("-> Best interaction: cID=%s, R2=%.6f"%(bestInteractionModel["cluster"], bestInteractionModel["R2"]))
        r2BestInt = bestInteractionModel["R2"]
        improvement = r2BestInt - r2NoInt 
        errorRedux = (r2BestInt-r2NoInt)/(1-r2NoInt)
        if onlyBestInteraction:
            data["Substrate"].append(s)
            data["R2"].append(bestInteractionModel["R2"])
            data["Model"].append("Best interaction fit")
            data["Rel. model improvement"].append(errorRedux)
            data["Abs. error reduction"].append(improvement)
        print("Improvement wrt no interaction: %.6f -> %.6f (abs. improvement: +%.6f, rel. error red.: %.6f)"%(r2NoInt, r2BestInt, improvement, errorRedux))
    
    data = pd.DataFrame(data)
    def extractInteractions(x):
        if x=="No interaction":
            return []
        y = x.replace("cluster","")
        y = y.replace("(","").replace(")","").replace("'","").replace(",","")
        y = y.split()
        return y
    
        
    # Plot R2 comparison
    fig, ax = plt.subplots()
    if substrate != "all":
        # plot comparison of different inhibitor setups for one substrate
        assert(nInt in (1,2))
        data = data.loc[data["Substrate"]==substrate]
        mods = list(data["Model"])
        interactions = [extractInteractions(x) for x in data["Model"]]
        ix_nInt = np.array([len(x) in (0,nInt) and x != [substrate] for x in interactions])
        data = data.loc[ix_nInt]
        modellabs = [", ".join(extractInteractions(x)) for x in data["Model"]]
        data["Model"] = modellabs
        colors = [config.clusterColors[7][cid[:3]] if cid != "" else config.clusterColors[7][substrate] for cid in data["Model"]]
        sns.barplot(data = data, x="Model", y="R2", ax=ax, figure=fig, palette=colors)
        plt.title("%s uptake inhibition"%substrate)
        fig.set_size_inches(plotDimensions["table_small"])
    else:
        modellabs = [", ".join(extractInteractions(x)) for x in data["Model"]]
        data["Model"] = modellabs
        sns.barplot("Substrate", "R2", "Model", data, ax=ax, figure=fig)
        fig.set_size_inches(plotDimensions["table"])
    plt.legend(loc="lower right")
    plt.ylim((0,1))
    plt.tight_layout()
    
    figname = "Comparison_uptake_fits%s_R2"%("_"+str(substrate) if substrate else "")
    if not onlyBestInteraction:
        figname += "_all_interactions"
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)
        
    # Plot absolute error reduction comparison
    fig, ax = plt.subplots()
    print(data["Model"]!="")
    interactionDataOnly = data[data["Model"]!=""]
    
    if substrate != "all":
        colors = [to_rgb(config.clusterColors[7][cid[:3]]) for cid in interactionDataOnly["Model"]]
        sns.barplot(data=interactionDataOnly, x="Model", y="Abs. error reduction", ax=ax, figure=fig, palette=colors)
        plt.title("%s uptake inhibition"%substrate)
        fig.set_size_inches(plotDimensions["table_small"])
    else:
        sns.barplot(data=interactionDataOnly, x="Substrate", y="Abs. error reduction", hue="Model", ax=ax, figure=fig)
        fig.set_size_inches(plotDimensions["table"])
    print(len(data["Substrate"].unique()), np.nanmax(data["R2"]))
    if not onlyBestInteraction:
        plt.legend(loc="lower right")
    plt.tight_layout()
    
    figname = "Comparison_uptake_fits_AbsErr%s_R2"%("_"+str(substrate) if substrate else "")

    if not onlyBestInteraction:
        figname += "_all_interactions"
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)
    
    # Plot relative error reduction comparison
    fig, ax = plt.subplots()
    print(data["Model"]!="No interaction")
    interactionDataOnly = data[data["Model"]!=""]
    if substrate != "all":        
        colors = [to_rgb(config.clusterColors[7][cid[:3]]) for cid in interactionDataOnly["Model"]]
        sns.barplot(data=interactionDataOnly, x="Model", y="Rel. model improvement", ax=ax, figure=fig, palette=colors)
        plt.title("%s uptake inhibition"%substrate)
        fig.set_size_inches(plotDimensions["table_small"])    
    else:
        sns.barplot(interactionDataOnly, x="Substrate", y="Rel. model improvement", hue="Model", ax=ax, figure=fig)
        fig.set_size_inches(plotDimensions["table"])
    print(len(data["Substrate"].unique()), np.nanmax(data["R2"]))
    plt.ylim((0,1))
    plt.ylabel("Rel. error reduction $q$")
    plt.tight_layout()
    
    figname = "Comparison_uptake_fits_RelErr%s_R2"%("_"+str(substrate) if substrate else "")
    if not onlyBestInteraction:
        figname += "_all_interactions"
    figname = os.path.join(FIG_DIR, figname+".svg")
    plt.savefig(figname)
    print("Saved figure '%s'"%figname)

    if show:
        plt.show()
    else:
        plt.close("all")
    
    
def plotImprovementsMatrices(df, fitter, show=False):
    # Plot a matrix: affecting clusters vs. substrate
    df = df.loc[df["nrInteractions"] == 1]
    substrates = sorted(set(df["substrate"]))
    clusters = fitter.getClusters()
    cids = sorted(list(clusters))
    score_labs = {"dR2": r"$\Delta R^2$",
                  "rel. fit improvement": r"relative fit improvement $p$", 
                  "rel. error reduction": r"relative error reduction $q$"}
    for score in ["dR2", "rel. fit improvement", "rel. error reduction"]:
        Z = np.zeros((len(substrates), len(clusters)))
        for i, sid in enumerate(substrates):
            dfs = df.loc[df["substrate"] == sid]
            for j, cid in enumerate(cids):
                cixs = clusters[cid]
                six = fitter.getSubstrateIx(sid)
                if six in cixs:
                    continue
                dfsc = dfs.loc[dfs["interactions"] == (cid,)]
                Z[i,j] = np.array(dfsc[score])[0]
        fig, ax = plt.subplots(layout="constrained", figsize=plotDimensions["matrix"])
        cmap='magma'
        c = ax.imshow(Z, extent=[-0.5, len(cids)-0.5, -0.5,len(substrates)-0.5], cmap=cmap, aspect='auto')
        ax.set_yticks(ticks=list(range(0,len(substrates))), labels=reversed(substrates))
        ax.set_xticks(ticks=list(range(0,len(cids))), labels=[f"[{cid}]" for cid in cids])
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        for x in list(range(0,len(cids))):
            ax.plot([x+0.5, x+0.5], [-0.5, len(substrates)-0.5], color="xkcd:white", alpha=0.5, lw=0.7)
        for y in list(range(0,len(substrates))):
            ax.plot([-0.5, len(cids)-0.5], [y+0.5, y+0.5], color="xkcd:white", alpha=0.5, lw=0.7)
        ax.set_ylabel("Substrate (inhibited)")
        ax.set_xlabel("Cluster (inhibitor)")
        fig.colorbar(c, label=score_labs[score])
        figname = Path(FIG_DIR ) / f"score_matrix({score}).svg"
        fig.savefig(figname)
        print(f"Saved figure '{figname}'")
    if show:
        plt.show()
    else:
        plt.close("all")
    

def plotGlucoseTimeseries(fitter, results, show=False):
    p = {}
    pars = results["interactions"]["params"]["Glc"]
    p["Nag"], p["Man"] = pars[("Nag",)], pars[("Man",)]
    p[None] = results["uptake"]["params"]["mix11"]["Glc"]

    errs = {}
    errors = results["interactions"]["errors"]["Glc"]
    errs["Nag"], errs["Man"] = errors[("Nag",)], errors[("Man",)]
    errs[None] = results["uptake"]["errors"]["mix11"]["Glc"]
    errs[None]["r2"] = 1 - errs[None]["ssr"]/errs[None]["sst"]

    print(f"dR2(Man) = {errs['Man']['r2']-errs[None]['r2']}")
    print(f"dR2(Nag) = {errs['Nag']['r2']-errs[None]['r2']}")
    print("errors:")
    pp(errors)


    for inh in ["Man", "Nag", None]:
        fitter.setUptakeParams({"Glc": p[inh]})
        fitter.resetInteractions()
        if inh:
            fitter.setInteractions({"Glc":([{"a":p[inh]["a"][0], 
                                            "clusterIndices":p[inh]["clusterIndices"][0]}])})
            fitter.activateInteractions()
        rids = [f"F{i}" for i in range(1,5)]
        for rid in rids: 
            sol, rhs = fitter.generateSubstrateSolution("Glc", rid)

            fig, axes = plt.subplots(layout="constrained", figsize=plotDimensions["timeline"], nrows=2, sharex=True)
            ax, axr = axes

            iPolV = fitter.makeCDWinterpolation(rid)
            plotSubstrateRateSim(sol, rhs, [0], ["Glc"], axr, False, 
                                invert=True, odSpanInterpolation=iPolV)
            plotSubstrateCurveSim(sol, [0], ["Glc"], ax, False)
            experiment = fitter.getExperiment()
            plotSubstrateCurveData(experiment, rid, ["Glc"], ax, False, 
                                ls="", plot_obs_range="minmax")
            plotSubstrateRateData(experiment, rid, ["Glc"], axr, False, 
                                ls="", invert=True, plot_obs_range="minmax", t_offset=10)
            if inh:
                plotSubstrateCurveData(experiment, rid, [inh], ax, False, 
                                    ls="", plot_obs_range="minmax")

            ax.set_title(rid)
            ax.set_ylabel("Concentration [mM]")
            axr.set_ylabel("Biomass-specific \ndepletion [µmol/g/h]")
            trange = 0.0, 55.0
            axr.set_xlim(trange)
            ax.legend()
            axr.legend()
            axr.set_xlabel(r"time $t$ [h]")
            ax.plot(trange, [0, 0], "k--", lw=0.8, zorder=-2)

            interaction_str = f"{inh} inhibition" if inh else "no inhibition"
            outdir = FIG_DIR / "glucose_inhibition"
            os.makedirs(outdir, exist_ok=True) 
            fn = outdir / f"Depletion glucose [{rid}], {interaction_str}.svg"
            fig.savefig(fn)
            print(f"Saved fig '{fn}'")
    if show:
        plt.show()
    else:
        plt.close("all")
    


def fullFitTimeseries(fitter, results, show=False):
    p = {}
    res = results["full, refit with interactions"]
    pars = res["params"]
    errs = res["errors"]
    errs["r2"] = 1 - errs["ssr"]/errs["sst"]
    print("\n# Plotting full fit")
    print("params:")
    pp(pars)
    print("errors:")
    pp(errs)

    # Prepare fitter
    fitter.setUptakeParams(pars)
    fitter.setGrowthParams(pars)
    fitter.setV0(pars["V0"])
    fitter.resetInteractions()
    fitter.setInteractions(pars["interactions"])
    fitter.activateInteractions()
    rids = fitter.getRunIDs()

    subs_groups = [
        ("His", "Thr", "Ile", "Phe"),
        ("Glc", "Man", "Nag"),
        ("Leu", "Val", "Lys", "Trp"),
    ]    

    for rid in rids: 
        sol, rhs = fitter.generateFullSolution(rid, False)
        fig, axes = plt.subplots(layout="constrained", figsize=(10,5), nrows=4, ncols=2, sharex=True, sharey=False)
        for i, subs in enumerate(subs_groups):
            ax, axr = axes[(1-int(i/2))*2, 1-i%2], axes[(1-int(i/2))*2+1,1-i%2]
            subs_ix = [2+fitter.getSubstrateIx()[s] for s in subs]
            plotSubstrateRateSim(sol, rhs, subs_ix, subs, 
                                axr, False, invert=True, odSpanInterpolation=None)
            plotSubstrateCurveSim(sol, subs_ix, subs, ax, False)
            experiment = fitter.getExperiment()
            plotSubstrateCurveData(experiment, rid, subs, ax, False, 
                                ls="", plot_obs_range="minmax")
            plotSubstrateRateData(experiment, rid, subs, axr, False, 
                                ls="", invert=True, plot_obs_range="minmax", t_offset=10)
            ax.set_ylabel("Concent-\nration [mM]")
            axr.set_ylabel("Spec. deple-\ntion [µmol/g/h]")
            trange = 0.0, 55.0
            axr.set_xlim(trange)

        ax, axr = axes[0,0], axes[1,0]
        plotGrowthCurveSim(sol, False, ax, colorShade=None, label=None, plotE=True)
        plotGrowthCurveData(experiment, rid, False, ax, biomassIndicator="CDW", ls="", plot_obs_range="minmax")
        plotGrowthRateSim(sol, rhs, False, axr, plotE=False)
        plotGrowthRateData(experiment, rid, False, axr, biomassIndicator="CDW", ls="", plot_obs_range="minmax")
        ax.set_ylabel("CDW [g/L]")
        axr.set_ylabel("Specific\ngrowth [1/h]")

        for ax in axes.flatten():
            ax.set_xlabel("")
            ax.plot(trange, [0, 0], "k--", lw=0.8, zorder=-2)
        axes[3,0].set_xlabel(r"time $t$ [h]")
        axes[3,1].set_xlabel(r"time $t$ [h]")


        outdir = FIG_DIR / "full_fit_plots"
        os.makedirs(outdir, exist_ok=True) 
        fn = outdir / f"Full fit [{rid}], no legend.svg"
        fig.savefig(fn)
        print(f"Saved fig '{fn}'")
        for ax in axes.flatten():
            ax.legend()
        fn = outdir / f"Full fit [{rid}].svg"
        fig.savefig(fn)
        print(f"Saved fig '{fn}'")
    if show:
        plt.show()
    else:
        plt.close("all")
    


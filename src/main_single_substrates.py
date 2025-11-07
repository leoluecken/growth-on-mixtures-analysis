from matplotlib.text import Text
import pandas as pd
import numpy as np 
from scipy.optimize import minimize
import os
from pprint import pp
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.patches import Rectangle

import datainput
import defaults
import config
import utils
import fitting
import plotting
from config import EnergyYields, FIG_DIR, colors, substrate_colors
from plotting import plotSubstrateCurveData, plotGrowthCurveSim,\
    plotSubstrateCurveSim, plotGrowthCurveData


figdir = FIG_DIR / "single_substrate_fits"
if not figdir.exists():
    os.makedirs(figdir, exist_ok=True)


exp_growth_interval = dict(
    Glc = [20.0, 50.0],
    His = [40.0, 90.0],
    Leu = [20.0, 40.0],
    Lys = [40.0, 75.0],
    Man = [20.0, 45.0],
    Nag = [12.0, 25.0],
    Phe = [12.0, 23.0],
    Thr = [15.0, 25.0],
    Trp = [30.0, 50.0],
    Val = [25.0, 50.0],
    )


def collectFitResultsSingle(results, fitters, plot=False, show=False):
    substrates = sorted(results["uptake"]["params"].keys())
    data = {s:{} for s in substrates}
    for s in substrates:
        print("\n#",s)
        params = results["uptake"]["params"][s][s]
        errors = results["uptake"]["errors"][s][s]
        errors["R2"] = 1 - errors["ssr"]/errors["sst"]
        print(f"  params:")
        pp(params)
        print(f"  errors:")
        pp(errors)
        
        fitters[s].setUptakeParams(params)
        experiment = fitters[s].getExperiment()
        
        if plot:
            fig, ax = plt.subplots(layout="constrained", figsize=(4.5,2.2))
            ax2 = ax.twinx()
            ax3 = ax.twiny()
            ax2.set_zorder(2)
            ax3.set_zorder(3)
            for rid in fitters[s].getRunIDs():
                plotSubstrateCurveData(experiment, rid, [s], ax3, 
                                       aggregateSubstrates=False, ls=".", 
                                       logscale=False, clusterColor=None, labels=True)
                sln, f = fitters[s].generateSubstrateSolution(s,rid)
                plotSubstrateCurveSim(sln, substrateIndices=[0], substrates=[s], 
                                      ax=ax3, aggregateSubstrates=False, 
                                      logscale=False, growthParams=None)
                ax.set_ylim([-0.1, 17])
                plotGrowthCurveData(experiment, rid, plotTDA=False, ax=ax2, 
                                    ls=".--", logscale=False, labels=True)
                
                ax2.set_ylim([-0.02, 1.5])
                ax.set_title(f"{s}: R2 = {errors['R2']:g}")
                ax.set_xlabel("Time [h]")
                ax.set_ylabel(f"{s} [mM]")
                ax2.set_ylabel(f"CDW/scaled OD [g/l]")
                ax3.set_xticks([], [])
                
                # Plot grey underlay
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                # fix xlim and ylim
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                exp_range = exp_growth_interval[s]
                ax.add_patch(Rectangle((exp_range[0], ylim[0]), exp_range[1]-exp_range[0], ylim[1]-ylim[0],
                              edgecolor=None, facecolor="#dddddd"))
                            
            figname = figdir / f"fit_{s}.svg"
            fig.savefig(figname)
            print(f"Saved figure '{figname}'")
            if show:
                plt.show()
            else:
                plt.close(fig)

        runIDs = sorted(experiment._tSpan.keys())
        
        yE = params["yE"]
        # Maximal growth rate from fit
        mu_fit = params["mu"]
        K_fit = params["K"]
        affinity_fit = mu_fit/K_fit
        
        data[s]["mu_fit"] = mu_fit
        data[s]["K_fit"] = K_fit
        data[s]["affinity_fit"] = affinity_fit
        data[s]["rate_at_S10"] = mu_fit/(K_fit+10)
        data[s]["rate_at_S5"] = mu_fit/(K_fit+5)
        data[s]["rate_at_S1"] = mu_fit/(K_fit+1)
        data[s]["yE"] = yE
        
        # Calculate exponential growth rate in experiment during interval I
        I = exp_growth_interval[s]
        #    V(t2) = exp(mu*(t2-t1))*V(t1)
        #    V(t2) = exp(mu*(t2-t1))*V(t1)
        ix1 = np.argmax(experiment._tSpan["F1"] >= I[0])
        for rid in experiment._cdwSpan.keys():
            ix1 = max(ix1, np.argmax(~np.isnan(experiment._cdwSpan[rid])))
        ix2 = np.argmax(experiment._tSpan["F1"] >= I[1])-1
        t1 = experiment._tSpan["F1"][ix1]
        t2 = experiment._tSpan["F1"][ix2]
        f = lambda gamma: sum(np.abs(vspan[ix2] - vspan[ix1]*np.exp(gamma*(t2-t1))) for rid, vspan in experiment._cdwSpan.items())
        gamma0 = affinity_fit
        res = minimize(fun=f, x0=gamma0, method="Nelder-Mead")
        gamma_exp = res.x[0]
        data[s]["gamma_exp"] = gamma_exp
        
        # Bacterial growth efficiency:
        # BGE = (V(t2)-V(t1))/(S(t1)-S(t2))
        bge_exp = np.mean([(experiment._cdwSpan[rid][ix2] - experiment._cdwSpan[rid][ix1])
                      /(experiment._sSpan[rid][s][ix1] - experiment._sSpan[rid][s][ix2])
                      for rid in runIDs]) 
        data[s]["bge_exp"] = bge_exp
    print("\n# Collected single substrate growth data:")
    pp(data)
    
    return data


def collectFitResultsMix(results, improvement_threshold):
    # Find optimal fits (selecting best interactions)
    base_errors = results["uptake"]["errors"]["mix11"]
    base_pars   = results["uptake"]["params"]["mix11"]
    subs = sorted(base_errors.keys())
    int_errors = {s:results["interactions"]["errors"][s] for s in subs}
    int_pars = {s:results["interactions"]["params"][s] for s in subs}
   
    improvement, best_int = dict(), dict()
    for s in subs:
        baseR2 = 1 - base_errors[s]["ssr"]/base_errors[s]["sst"]
        # Find best R2 with interaction 
        intR2, int_ = baseR2, None
        for i, d in int_errors[s].items():
            if d["r2"] > intR2:
                intR2 = d["r2"]
                int_ = i
        improvement[s] = (intR2 - baseR2)/baseR2
        best_int[s] = int_
    pp(improvement)
    pp(best_int)
    
    opt_pars = dict()
    for s in subs:
        if improvement[s] > improvement_threshold:
            opt_pars[s] = int_pars[s][best_int[s]]
            opt_pars[s]["interaction"] = best_int[s]
        else:
            opt_pars[s] = base_pars[s]
            opt_pars[s]["interaction"] = None
    return opt_pars
        
    
def compare_single_and_mix(data_ss, data_mix):
    subs = sorted(data_ss.keys())
    aminos = ['His', 'Leu', 'Lys', 'Phe', 'Thr', 'Trp', 'Val']
    sugars = ['Glc', 'Nag', 'Man']
    
    #  Calculate parameters changes single substrate → mix
    #    mu, K, affinity, rate_at_10
    changes = defaultdict(dict)
    for s in subs:
        changes[s]["mu"] = data_mix[s]["mu"]/data_ss[s]["mu_fit"]
        changes[s]["K"] = data_mix[s]["K"]/data_ss[s]["K_fit"]
        affinity = data_mix[s]["mu"]/data_mix[s]["K"]
        changes[s]["affinity"] = affinity/data_ss[s]["affinity_fit"]
        rate_at_S10 = data_mix[s]["mu"]/(data_mix[s]["K"] + 10)
        changes[s]["rate_at_S10"] = rate_at_S10/data_ss[s]["rate_at_S10"]
        rate_at_S5 = data_mix[s]["mu"]/(data_mix[s]["K"] + 5)
        changes[s]["rate_at_S5"] = rate_at_S5/data_ss[s]["rate_at_S5"]
        rate_at_S1 = data_mix[s]["mu"]/(data_mix[s]["K"] + 1)
        changes[s]["rate_at_S1"] = rate_at_S1/data_ss[s]["rate_at_S1"]
    changes_rateS10 = np.array([changes[s]["rate_at_S10"] for s in subs])
    changes_rateS5 = np.array([changes[s]["rate_at_S5"] for s in subs])
    changes_rateS1 = np.array([changes[s]["rate_at_S1"] for s in subs])
    changes_affinity = np.array([changes[s]["affinity"] for s in subs])
    changes_mu = np.array([changes[s]["mu"] for s in subs])
    changes_K= np.array([changes[s]["K"] for s in subs])
    pp(changes)
    meanKmix = np.mean([data_mix[s]['K'] for s in subs])
    meanKss = np.mean([data_ss[s]['K_fit'] for s in subs])
    mean_mumix = np.mean([data_mix[s]['mu'] for s in subs])
    mean_muss = np.mean([data_ss[s]['mu_fit'] for s in subs])
    affinities_mix = [data_mix[s]["mu"]/data_mix[s]["K"] for s in subs]
    affinities_ss = [data_ss[s]["mu_fit"]/data_ss[s]["K_fit"] for s in subs]
    print(f"Mean K in mix: {meanKmix}")
    print(f"Mean K indiv: {meanKss}")
    print(f"Mean mu in mix: {mean_mumix}")
    print(f"Mean mu indiv: {mean_muss}")
    print(f"affinities in mix: {affinities_mix}")
    print(f"affinities indiv: {affinities_ss}")
    print(f"change affinities: {np.array(affinities_mix)/affinities_ss}")
    
    plot_pairplots = False
    if plot_pairplots:
        fig, axes = plt.subplots(nrows=3, ncols=8, layout="constrained", figsize=(16,6))
        axBGE, axGamma, axY = axes
        bge = np.array([data_ss[s]["bge_exp"] for s in subs])
        gamma = np.array([data_ss[s]["gamma_exp"] for s in subs])
        yE = np.array([data_ss[s]["yE"] for s in subs])
        
        mus_ss = np.array([data_ss[s]["mu_fit"] for s in subs]) 
        mus_mix = np.array([data_mix[s]["mu"] for s in subs] )

        ix = [s in subs for s in subs]
        for i, kv in enumerate([
            ("$\\mu_{s}$",mus_ss), ("$\\mu_{mix}$",mus_mix),
            ("$\\Delta K$",changes_K), ("$\\Delta \\mu$",changes_mu),
            ("$\\Delta aff.$",changes_affinity), ("$\\Delta r_{1}$",changes_rateS1),
            ("$\\Delta r_{5}$",changes_rateS5), ("$\\Delta r_{10}$",changes_rateS10),]):
            k, v = kv
            axBGE[i].plot(bge[ix], v[ix], "x", label=k)
            axGamma[i].plot(gamma[ix], v[ix], "x", label=k)
            axY[i].plot(yE[ix], v[ix], "x", label=k)
            axBGE[i].plot(axBGE[i].get_xlim(), [0,0], "k--", lw=0.5, zorder=-1)
            axGamma[i].plot(axGamma[i].get_xlim(), [0,0], "k--", lw=0.5, zorder=-1)
            axY[i].plot(axY[i].get_xlim(), [0,0], "k--", lw=0.5, zorder=-1)
            axBGE[i].set_xlabel("BGE")
            axGamma[i].set_xlabel("gamma")
            axY[i].set_xlabel("yE")
            axBGE[i].set_ylabel(k)
            axGamma[i].set_ylabel(k)
            axY[i].set_ylabel(k)
        plt.show()
    
    symbol = {
        "gamma_exp":"Growth rate $\\gamma$", 
        "bge_exp":"BGE $\\eta$", 
        "yE":"Energy yield $y_E$",
        "dmu":"$\\Delta\\mu$", 
        "dr1":"$\\Delta r_1$", 
        "dK":"$\\Delta K$"
    }
    
    plot_sugar_comparison = True
    if plot_sugar_comparison:
        df_sugars = dict(substrate=[], value=[], var=[])
        for s in ("Glc", "Nag", "Man"):
            for k in ("affinity_fit", "rate_at_S10"):
                df_sugars["substrate"].append(s)
                df_sugars["var"].append(k)
                df_sugars["value"].append(data_ss[s][k])
            for k in ("gamma_exp", "bge_exp", "yE"):
                df_sugars["substrate"].append(s)
                df_sugars["var"].append(k)
                df_sugars["value"].append(data_ss[s][k])
           
        df_sugars = pd.DataFrame(df_sugars)
        fig, axes = plt.subplots(layout="constrained", nrows=3, figsize=(2.5,4.0))
        for ax, k in zip(axes, ("gamma_exp", "bge_exp", "yE")):
            ix = df_sugars.groupby("var").groups[k]
            dfk = df_sugars.iloc[ix,:]
            sns.barplot(dfk, x="var", y="value", hue="substrate", ax=ax,
                        palette=[substrate_colors[s] for s in ("Glc", "Nag", "Man")])
            ax.set_ylabel(symbol[k])
            ax.set_xticks([-0.25, 0.0, 0.25], ["Glc", "Nag", "Man"])
            # if k!="gamma_exp":
            ax.legend().remove()
        fig.suptitle("Isolated sugar utilization")
        figname = figdir / "comparison_sugar_characteristics.svg"
        fig.savefig(figname)
        print(f"\nSaved fig '{figname}'")
        plt.show()
    
    
    
    plot_amino_comparison = True
    if plot_amino_comparison:
        df_aminos = defaultdict(list)
        for s in subs:
            df_aminos["substrate"].append(s)
            df_aminos["dmu"].append(changes[s]["mu"]-1)
            df_aminos["dr10"].append(changes[s]["rate_at_S10"]-1)
            df_aminos["dr5"].append(changes[s]["rate_at_S5"]-1)
            df_aminos["dr1"].append(changes[s]["rate_at_S1"]-1)
            df_aminos["dK"].append(changes[s]["K"])
            for k in ("gamma_exp", "bge_exp", "yE"):
                df_aminos[k].append(data_ss[s][k])
        fig, axes = plt.subplots(layout="constrained", nrows=3, ncols=3, figsize=(8,7))
        for axs, k0 in zip(axes, ("dmu", "dK", "dr1")):
            for ax, k in zip(axs, ("gamma_exp", "bge_exp", "yE")):
                sns.scatterplot(df_aminos, x=k, y=k0, hue="substrate", ax=ax, palette=[substrate_colors[s] for s in subs])
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                xr, yr = xlim[1]-xlim[0], ylim[1]-ylim[0] 
                for x, y, s in zip(df_aminos[k], df_aminos[k0], df_aminos["substrate"]):
                    if s in sugars:
                        ax.text(x=x+0.03*xr, y=y+0.03*yr, s=s, color="xkcd:red")
                    else:
                        ax.text(x=x+0.03*xr, y=y+0.03*yr, s=s)
                ax.set_xlim((xlim[0], xlim[1]+0.2*xr))
                ax.plot(ax.get_xlim(), [0,0], "k--", lw=0.5, zorder=-1)
                ax.plot(ax.get_xlim(), [-1,-1], "k--", lw=0.5, zorder=-1)
                ax.legend().remove()
                ax.set_xlabel(symbol[k])
                ax.set_ylabel(symbol[k0])
        fig.suptitle("Changes in amino acid utilization")
        figname = figdir / "comparison_amino_characteristics.svg"
        fig.savefig(figname)
        print(f"\nSaved fig '{figname}'")
        plt.show()
    print("")
    
def main():
    singleSubstrateExperiment = datainput.loadSingleSubstrateData()
    singleSubstrates = utils.orderSubstrates(singleSubstrateExperiment.keys(), None, config.substrateOrderKey)
    
    # Init fitters
    singleSubstrateFitter = {}
    for s in singleSubstrates:
        exp = singleSubstrateExperiment[s]
        params = defaults.makeParams([s], len(exp._runIDs))
        params["fitK"] = True
        singleSubstrateFitter[s] = fitting.DEBFitter(exp, params)
        
    # Collect Single substrate data
    results_fn = config.FIT_DIR / "results_single_substrates.pickle"
    results={}
    if not results_fn.exists():
        utils.saveState(results, results_fn)
    
    skipSingleUptakeFit = True
    if not skipSingleUptakeFit:
        # Fit substrate uptake parameters
        print("\nFitting uptake parameters.")
        # Nr of LHS presampling parameter guesses
        LHS_presampling = 0 
        # LHS_presampling = 100 
        uptakeParams, uptakeErrors  = fitting.fitUptakeParameterBatch(singleSubstrateFitter, singleSubstrates, LHS_presampling=LHS_presampling)
        results["uptake"] = {}
        results["uptake"]["params"] = uptakeParams
        results["uptake"]["errors"] = uptakeErrors        
        print("results")
        pp(results)
        utils.saveState(results, results_fn)
    results_loaded = utils.loadState(results_fn)
    
    # Debug
    print("results")
    pp(results)
    print("results loaded:")
    pp(results_loaded)
    
    if results == {}:
        results = results_loaded
    
    data_ss = collectFitResultsSingle(results, singleSubstrateFitter, plot=True)

    # Collect Mixture data
    results_fn = config.FIT_DIR / "results_mix11_all_runs.pickle"
    if not results_fn.exists():
        raise Exception("Please generate mixture fit first, see main_P_inhibens_fit.py")
    else:
        results = utils.loadState(results_fn)
    data_mix = collectFitResultsMix(results, improvement_threshold = 0.03)
    
    compare_single_and_mix(data_ss, data_mix)

    
if __name__ == '__main__':
    main()

    
    
    
from collections import defaultdict
import os
import re
from pprint import pp
import pickle
import toml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import config
from main_fit_synthetic_data import prepareFitter, get_scan_variant_description
from config import FIG_DIR

figdir_base = FIG_DIR / "synthetic_data_fits"
if not figdir_base.exists():
    os.makedirs(figdir_base, exist_ok=True)


def get_model_trajectories(fit_res, predicted_interactions, data_fn):
    expID = list(fit_res["uptake"]["params"].keys())[0]
    multiSubstrateFitter = prepareFitter(data_fn, experimentID=expID)
    subs = sorted(fit_res["uptake"]["params"][expID].keys())
    
    # Set up parameters
    params = defaultdict(dict)
    interactions = defaultdict(list)
    for s in subs:
        ints = predicted_interactions[s]
        if ints is None:
            params[s] = fit_res["uptake"]["params"][expID][s]
        else:
            params[s] = fit_res["interactions"]["params"][s][ints]
            interactions[s].append(dict(a=params[s]["a"], clusterIndices=params[s]["clusterIndices"]))
    # multiSubstrateFitter.setInteractions(interactions)
    multiSubstrateFitter.setUptakeParams(params)
    multiSubstrateFitter.activateInteractions()
    
    # Generate trajectories
    slns = defaultdict(dict)
    for sid in subs:
        for rid in multiSubstrateFitter.getRunIDs():
            res, ds = multiSubstrateFitter.generateSubstrateSolution(sid, rid)
            tspan = np.arange(res.t[0], res.t[-1], 0.25)
            sspan = res.sol(tspan)[0]
            dsspan = [ds(t, s) for t,s in zip(tspan, sspan)]
            cdw_span = multiSubstrateFitter._dynamics._iPolV[rid](tspan)
            slns[sid][rid] = dict(t=tspan, s=sspan, ds=dsspan, cdw=cdw_span)
            
    
    # Collect in DataFrame
    df, df_rates = defaultdict(list), defaultdict(list)
    for sid in subs:
        for rid in multiSubstrateFitter.getRunIDs():
            nt = len(slns[sid][rid]["t"])
            df["t"].extend(slns[sid][rid]["t"])
            df["concentration"].extend(slns[sid][rid]["s"])
            df["substrate"].extend([sid]*nt)
            df["cdw"].extend(slns[sid][rid]["cdw"])
            df["rid"].extend([rid]*nt)
            
            df_rates["t"].extend(tspan)
            df_rates["uptake rate"].extend(slns[sid][rid]["ds"])
            df_rates["substrate"].extend([sid]*nt)
            df_rates["rid"].extend([rid]*nt)
    df, df_rates = pd.DataFrame(df), pd.DataFrame(df_rates) 
    return df, df_rates
            

def get_fit_errors(fit_res, predicted_interactions):
    expID = list(fit_res["uptake"]["params"].keys())[0]
    subs = sorted(fit_res["uptake"]["params"][expID].keys())

    # Set up parameters
    errors = defaultdict(dict)
    for s in subs:
        ints = predicted_interactions[s]
        if ints is None:
            errors[s] = fit_res["uptake"]["errors"][expID][s]
        else:
            errors[s] = fit_res["interactions"]["errors"][s][ints]
        if "r2" not in errors[s]:
            errors[s]["r2"] = 1 - errors[s]["ssr"]/errors[s]["sst"]
    return errors


def getClusters(data_fn, similarity_threshold=0.075):
    multiSubstrateFitter = prepareFitter(data_fn, experimentID=None)
    multiSubstrateFitter.buildSubstrateClusters(figname=None, 
                                                similarity_threshold=similarity_threshold)
    clusters = multiSubstrateFitter.getClusters()
    clustermap = dict()
    for s in multiSubstrateFitter.getSubstrates():
        six = multiSubstrateFitter.getSubstrateIx(s)
        for sid, cixs in clusters.items():
            if six in cixs:
                clustermap[s] = sid
    return clustermap


def plot_fit(data_fn, param_fn, fit_fn, clusters, predicted_interactions, 
             fig_fn, include_a_values=False, show=False):
    orig_params = toml.load(param_fn)    
    df_exp = pd.read_csv(data_fn)
    subs = [s for s in df_exp.columns if s[0]=="S"]
    
    # Make rates
    rid_groups = df_exp.groupby("rid").groups
    runIDs = sorted(rid_groups.keys())
    dfs_rates = defaultdict(dict)
    for rid in runIDs:
        dfg = df_exp.iloc[rid_groups[rid],:]
        rates = {s:(np.array(dfg[s])[1:]-np.array(dfg[s])[:-1])
                 /(np.array(dfg["t"])[1:]-np.array(dfg["t"])[:-1]) 
                 for s in dfg.columns if not s in {'Unnamed: 0', "rid"}}
        rates["t"] = (np.array(dfg["t"])[1:]+np.array(dfg["t"])[:-1])/2
        dfr = pd.DataFrame(rates)
        dfr["rid"] = rid
        dfs_rates[rid] = dfr
    
    true_interactions = {s:orig_params["interactions"].get(s, None) for s in subs}
    for s in sorted(true_interactions.keys()):
        if true_interactions[s] is not None:
            if include_a_values:
                true_interactions[s] = tuple(f"S{int(i['clusterIndices'][0]):02d}"
                                             + f" [a={float(i['a']):.1f}]" 
                                             for i in true_interactions[s])
            else:
                true_interactions[s] = tuple(f"S{int(i['clusterIndices'][0]):02d}"
                                             for i in true_interactions[s])
            
    with open(fit_fn, "rb") as f:
        fit_res = pickle.load(f)
        
    predicted_ints = predicted_interactions.copy()
    for s, ints in predicted_interactions.items():
        if ints:
            if include_a_values:
                a = tuple(fit_res["interactions"]["params"][s][ints]["a"])
                predicted_ints[s] = tuple(f"{i} [a={va:.1g}]" for i, va in zip(ints, a))
            else:
                predicted_ints[s] = tuple(f"{i}" for i in ints)
        
    df_model, df_rates_model = get_model_trajectories(fit_res, predicted_interactions, data_fn)
    fit_errors = get_fit_errors(fit_res, predicted_interactions)

    dfs = df_exp.melt(id_vars=["CDW", "t", "rid"], value_vars=subs, 
                  var_name="substrate", value_name="concentration")
    dfs_rates = {rid:dfs_rates[rid].melt(id_vars=["CDW", "t", "rid"], value_vars=subs, 
                var_name="substrate", value_name="uptake rate") for rid in runIDs}
    dfs_rate_groups = {rid:dfs_rates[rid].groupby("substrate").groups for rid in runIDs}
    gg = dfs.groupby("substrate").groups
    gg_model = df_model.groupby("substrate").groups
    gg_rates_model = df_rates_model.groupby("substrate").groups
    fig, axes = plt.subplots(ncols=int(np.ceil(np.round(len(gg)/2,1))), 
                             nrows=2, layout="constrained", figsize=(10,5))
    fig.suptitle(f"clusters: {clusters}")
    axes = {s:ax for s, ax in zip(gg,axes.flatten())} 
    for s, ix in gg.items():
        dfg = dfs.iloc[ix,:]
        ax = axes[s]
        ax2 = ax.twinx()
        
        # Plot concentrations
        dfg_model = df_model.iloc[gg_model[s],:]
        sns.scatterplot(dfg, x="t", y="concentration", hue="rid", marker="x", ax=ax, legend=None)
        sns.lineplot(dfg_model, x="t", y="concentration", hue="rid", ax=ax, legend=None)
        sns.lineplot(dfg_model, x="t", y="cdw", hue="rid", ax=ax, legend=None, ls="--") #, palette="binary")
        
        # Plot rates
        dfr = pd.concat([dfs_rates[rid].iloc[dfs_rate_groups[rid][s],:] for rid in runIDs])
        dfr_model = df_rates_model.iloc[gg_rates_model[s],:]
        sns.scatterplot(dfr, x="t", y="uptake rate", hue="rid", marker="x", ax=ax2, legend=None)
        sns.lineplot(dfr_model, x="t", y="uptake rate", hue="rid", ax=ax2, legend=None)
        ylim1, ylim2 = ax.get_ylim(), ax2.get_ylim()
        
        miny_model = np.nanmin(dfr_model["uptake rate"])
        miny_exp = np.nanquantile(dfr["uptake rate"], 0.2)
        ylim2 = (min(miny_exp, miny_model)*1.1, ylim2[1])
        
        ax.set_ylim([-ylim1[1]*1.2, ylim1[1]])
        ax2.set_ylim([ylim2[0], -ylim2[0]*1.1])
        xlim1, xlim2 = ax.get_xlim(), ax2.get_xlim()
        ax.set_xlim(xlim1), ax2.set_xlim(xlim2)
        ax.plot(xlim1, [0,0], "k--", lw=0.5, zorder=-1)
        ax2.plot(xlim2, [0,0], "k--", lw=0.5, zorder=-1)
        
        ax.set_title(f"{s}: R2={fit_errors[s]['r2']:g}\n(predicted → {predicted_ints[s]}, true → {true_interactions[s]})")
    fig.savefig(fig_fn)
    print(f"Saved fig '{fig_fn}'")
    if show:
        plt.show()
    else:
        plt.close(fig)


def collect_data(seed, param_fn, fit_fn):
    with open(fit_fn, "rb") as f:
        fit_result = pickle.load(f)    
    orig_params = toml.load(param_fn)
    print("\n# Original parameter:")
    pp(orig_params)
    
    
    df = {
        "seed":[],
        "substrate":[],
        "mu":[],
        "K":[],
        "mu_fit":[],
        "K_fit":[],
        "interaction0":[],
        "interaction1":[],
        "a0":[],
        "a1":[],
        "a0_fit":[],
        "a1_fit":[],
        "sst":[],
        "ssr":[],
        "R2":[],
        }
    
    params_no_interaction = fit_result["uptake"]["params"][f"exp{seed}"]
    errors_no_interaction = fit_result["uptake"]["errors"][f"exp{seed}"]
    
    best_R2 = {"0int":{}, "1int":{}, "2ints":{}}
    for s in params_no_interaction:
        df["seed"].append(seed)
        df["substrate"].append(s)
        df["mu"].append(orig_params[s]["mu"])
        df["K"].append(orig_params[s]["K"])
        df["mu_fit"].append(params_no_interaction[s]["mu"])
        df["K_fit"].append(params_no_interaction[s]["K"])
        df["interaction0"].append(None)
        df["interaction1"].append(None)
        df["a0"].append(None)
        df["a1"].append(None)
        df["a0_fit"].append(None)
        df["a1_fit"].append(None)
        df["sst"].append(errors_no_interaction[s]["sst"])
        df["ssr"].append(errors_no_interaction[s]["ssr"])
        R2 = 1 - df["ssr"][-1]/df["sst"][-1] 
        df["R2"].append(R2)
        best_R2["0int"][s] = R2
    
    params_interaction = fit_result["interactions"]["params"]
    errors_interaction = fit_result["interactions"]["errors"]
        
    for s0 in params_interaction:
        for ints in params_interaction[s0]:
            df["seed"].append(seed)
            df["substrate"].append(s0)
            df["mu"].append(orig_params[s0]["mu"])
            df["K"].append(orig_params[s0]["K"])
            df["mu_fit"].append(params_interaction[s0][ints]["mu"])
            df["K_fit"].append(params_interaction[s0][ints]["K"])
            df["sst"].append(errors_interaction[s0][ints]["sst"])
            df["ssr"].append(errors_interaction[s0][ints]["ssr"])
            R2 = 1 - df["ssr"][-1]/df["sst"][-1] 
            df["R2"].append(R2)
            
            a = params_interaction[s0][ints]["a"]
            orig_a = np.zeros(len(ints))
            orig_ints = orig_params["interactions"].get(s0,[])
            for i in orig_ints:
                orig_clusterIndices = i["clusterIndices"]
                assert(len(orig_clusterIndices) == 1)
                six = int(orig_clusterIndices[0])
                orig_subs = f"S{six:02d}"
                if orig_subs in ints:
                    int_ix = np.where([orig_subs == ii for ii in ints])[0]
                    orig_a[int_ix] = float(i["a"])
                else:
                    continue
                    
            df["a0"].append(orig_a[0])
            df["interaction0"].append(ints[0])
            df["a0_fit"].append(a[0])

            if len(ints) == 2:
                df["a1"].append(orig_a[1])
                df["interaction1"].append(ints[1])
                df["a1_fit"].append(a[1])
                best_R2["2ints"].setdefault(s0, (None, -np.inf))
                if best_R2["2ints"][s0][1] < R2:
                    best_R2["2ints"][s0] = (ints, R2)
            else:
                df["a1"].append(None)
                df["interaction1"].append(None)
                df["a1_fit"].append(None)
                best_R2["1int"].setdefault(s0, (None, -np.inf))
                if best_R2["1int"][s0][1] < R2:
                    best_R2["1int"][s0] = (ints, R2)
    df = pd.DataFrame(df)
    return df, best_R2, orig_params



def predict_interactions(best_R2, improvement_th, use_error_reduction):
    subs = sorted(best_R2["0int"].keys())
    
    base_R2 = {s:best_R2["0int"][s] for s in subs}
    int1_R2 = {s:best_R2["1int"][s][1] for s in subs}
    int2_R2 = {s:best_R2["2ints"][s][1] for s in subs if s in best_R2["2ints"]}
    
    if use_error_reduction:
        # Percentual error decrease (error = 1-R2)
        q1 = {s:(int1_R2[s]-base_R2[s])/(1-base_R2[s]) for s in subs}
        q2 = {s:(int2_R2[s]-int1_R2[s])/(1-int1_R2[s]) for s in subs if s in int2_R2}
    else:
        # Percentual R2 increase    
        q1 = {s:(int1_R2[s]-base_R2[s])/base_R2[s] for s in subs}
        q2 = {s:(int2_R2[s]-int1_R2[s])/int1_R2[s] for s in subs if s in int2_R2}
        
    print("\n# Improvement base -> 1 interaction:")
    pp(q1)

    print("\n# Improvement 1 interaction -> 2 interactions:")
    pp(q2)
    
    predicted = {}
    for s in subs:
        if (s in q2) and (q2[s] > improvement_th):
            predicted[s] = best_R2["2ints"][s][0]
        elif q1[s] > improvement_th:
            predicted[s] = best_R2["1int"][s][0]
        else:
            predicted[s] = None
    
    print(f"\n# Predicted interactions (th={improvement_th:g}):")
    pp(predicted)
    
    return predicted
    

def performance_scores(predicted_interactions, orig_params, clusters):
    subs = sorted(predicted_interactions.keys())
    
    present = set()
    for s0, ints in orig_params["interactions"].items():
        for i in ints:
            s1 = f"S{int(i['clusterIndices'][0]):02d}"
            present.add((s0, clusters[s1]))
    predicted = set()
    for s0 in subs:
        ints = predicted_interactions[s0]
        if ints is not None:
            for s1 in ints:
                predicted.add((s0,clusters[s1]))
    ures = np.unique_counts([clusters[s] for s in subs])
    singletons = ures.values[np.where(ures.counts)[0]]
    
    tp = sorted(present.intersection(predicted))
    tp_singleton = set(i for i in tp if clusters[i[1]] in singletons)
    fp = sorted(predicted.difference(present))
    fn = sorted(present.difference(predicted))
    
    if len(present)==0:
        recall = 1
    else:
        recall = len(tp)/len(present)

    if len(tp) == 0:
        if len(fp) == 0:
            precision = 1.0
        else:
            precision = 0.0
    else:
        precision = len(tp)/(len(tp) + len(fp))
    if precision*recall==0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)
    
    print("\n# True interactions:")
    pp(sorted(present))
    return (f1, recall, precision, len(tp), 
            len(tp_singleton), len(fp), len(fn))
    
    
def assess_fit(seed, param_fn, fit_fn, clusters, improvement_threshold, use_error_reduction):
    print(f"Assessing fit '{seed}'")
    df, best_R2, orig_params = collect_data(seed, param_fn, fit_fn)
    
    predicted_interactions = predict_interactions(best_R2, improvement_threshold, use_error_reduction)
    f1, recall, precision, tps, tps_singleton, fps, fns = performance_scores(predicted_interactions, orig_params, clusters)
    
    score = pd.DataFrame(dict(seed=[seed], f1=[f1], recall=[recall], precision=[precision],
                              tps=[tps], tps_singleton=[tps_singleton], fps=[fps], fns=[fns]))
    print("\n# Score:")
    pp(score)
    return df, score, predicted_interactions


def run(datadir, resultsdir, improvement_threshold, scan_variant, 
        plot_data, use_error_reduction, score_aggregation, load=True):

    score_measure = "err_reduction" if use_error_reduction else "fit_improvement"
    save_fn = FIG_DIR / "synthetic_data_fits" / f"scan{scan_variant}_{score_measure}.pickle"

    if load and save_fn.exists():
        with open(save_fn, "rb") as f:
            df_scores_aggr = pickle.load(f)
    else:
        desc = get_scan_variant_description(scan_variant)
        
        param_fns = [fn for fn in datadir.iterdir() if fn.name.split(".")[-1]=="toml"]
        data_fns = [fn for fn in datadir.iterdir() if fn.name.split(".")[-1]=="csv"]
        fit_fns = [fn for fn in resultsdir.iterdir() if fn.name.split(".")[-1]=="pickle"]
        
        p = re.compile(r"[a-z_]+([0-9-]+)_seed([0-9]+)\.[a-z]+")
        param_fns = {re.match(p, fn.name).group(2):fn for fn in param_fns}
        data_fns = {re.match(p, fn.name).group(2):fn for fn in data_fns}
        assert(set(param_fns) == set(data_fns))
        
        pfit = re.compile(r"results_synthetic_data_seed([0-9]+)\.pickle")
        fit_fns = {re.match(pfit, fn.name).group(1):fn for fn in fit_fns}
        
        seeds = set(param_fns).intersection(fit_fns)
        print(f"Found results for {len(seeds)} seeds:")
        pp(seeds)
        
        parcombis = {re.match(p, fn.name).group(2):re.match(p, fn.name).group(1).split("-")[0] for fn in data_fns.values()}
        parcombis = {seed:desc["ticks"][int(p)] for seed, p in parcombis.items()}
        
        missing = set(param_fns).difference(fit_fns)
        print("Seeds for which no results were found:")
        pp(missing)
        
        dfs, scores, interactions = {}, {}, {}
        for seed in sorted(seeds):
            clusters = getClusters(data_fns[seed])
            df, score, predicted_interactions = assess_fit(seed, param_fns[seed], fit_fns[seed], 
                                                        clusters, improvement_threshold,
                                                        use_error_reduction)
            score["parcombi"] = [parcombis[seed]]
            score["seed"] = [seed]
            dfs[seed] = df
            scores[seed] = score
            interactions[seed] = predicted_interactions        
            if plot_data:
                fig_fn = data_fns[seed].parent / (".".join(data_fns[seed].name.split(".")[:-1]) + ".svg")
                plot_fit(data_fns[seed], param_fns[seed], fit_fns[seed], 
                        clusters, predicted_interactions, fig_fn)
        df_all = pd.concat(dfs.values())
        df_scores = pd.concat(scores.values())
        df_scores.index = range(len(df_scores)) 
        print(df_scores)
        
        # Get per par-combi scores
        parcombis = sorted(set(df_scores["parcombi"]))
        gg = df_scores.groupby("parcombi").groups
        df_scores_aggr = defaultdict(list)
        for pc in parcombis:
            ix = gg[pc]
            dfg = df_scores.iloc[ix,:]
            df_scores_aggr["parcombi"].append(pc)
            fps, tps, fns = np.sum(dfg["fps"]), np.sum(dfg["tps"]), np.sum(dfg["fns"])
            df_scores_aggr["fps"].append(fps)
            df_scores_aggr["fns"].append(fns)
            df_scores_aggr["tps"].append(tps)
            if score_aggregation=="total":
                recall = tps/(fns + tps)
                if tps==0:
                    precision = 0.0
                else:
                    precision = tps/(fps + tps)
                if precision*recall==0:
                    f1 = 0
                else:
                    f1 = 2*precision*recall/(precision+recall) 
            elif score_aggregation=="mean":
                recall = np.nanmean(dfg["recall"])
                precision = np.nanmean(dfg["precision"])
                f1 = np.nanmean(dfg["f1"])
            else:
                raise(f"Aggregation type score_aggregation='{score_aggregation}' not implemented")
            df_scores_aggr["recall"].append(recall)
            df_scores_aggr["precision"].append(precision)
            df_scores_aggr["f1"].append(f1)
            df_scores_aggr["n"].append(len(ix))
        
        df_scores_aggr = pd.DataFrame(df_scores_aggr)
        with open(save_fn, "wb") as f:
            pickle.dump(df_scores_aggr, f)
    print(f"\nData in '{save_fn.name}':")
    print(df_scores_aggr)
    df_scores_aggr = df_scores_aggr.melt(id_vars=["parcombi"], value_vars=["f1", "recall", "precision"],
                             var_name="measure", value_name="mean score")
    return df_scores_aggr

    
def plotScores(dfs, scan_variant, use_error_reduction, score_aggregation, show=False):
    figdir = figdir_base  / f"scan{scan_variant}"
    if not figdir.exists():
        os.makedirs(figdir, exist_ok=True)
        
    colors = dict(
        f1=["#001F3F", "#3A6D8C", "#6A9AB0", "#8AAAC0", "#AAAAC0"],
        precision=["#001F3F", "#3A6D8C", "#6A9AB0", "#8AAAC0", "#AAAAC0"],
        recall=["#001F3F", "#3A6D8C", "#6A9AB0", "#8AAAC0", "#AAAAC0"],
        )
    lss =  dict(
        f1="-",
        precision="-",
        recall="-"
        )
        
    desc = get_scan_variant_description(scan_variant)
    fig, axes = plt.subplots(layout="constrained", nrows=3, figsize=(4,5))
    improvement_thresholds = sorted(dfs.keys())
    measures = ["f1", "recall", "precision"]
    for i, improvement_threshold in enumerate(improvement_thresholds):
        df = dfs[improvement_threshold]
        gg = df.groupby("measure").groups 
        leg = i==0
        for ax, measure in zip(axes, measures):
            ix = gg[measure]
            ls = lss[measure]
            dfg = df.iloc[ix,:]
            sns.lineplot(dfg, x="parcombi", y="mean score", ax=ax, legend=leg, ls=ls, color=colors[measure][i], 
                         label=f"th={improvement_threshold:g}")
            sns.scatterplot(dfg, x="parcombi", y="mean score", ax=ax, color=colors[measure][i])
            ax.set_ylim((-0.05, 1.1))
            ax.set_ylabel(measure)
            ax.set_xlabel(desc["label"])
            if i == 0:
                ax.set_xlim(ax.get_xlim())
                ax.plot(ax.get_xlim(), [0.5, 0.5], color="grey", 
                        lw=0.5, ls="--", zorder=-2)
    axes[0].legend()
            
    ax.set_ylim((-0.05,1.1))
    fig.suptitle(f"Scan {scan_variant} [{desc['title']}]")
    ths = ["%g"%th for th in improvement_thresholds]
    indicator_str = "err_reduction" if use_error_reduction else "fit_improvement"
    figname = figdir / f"performance_{score_aggregation}_scores_scan{scan_variant}_th{ths}_indicator_{indicator_str}.svg"
    fig.savefig(figname)
    print(f"Saved fig '{figname}'")
    if show:
        plt.show()
    else:
        plt.close("all")
    

if __name__ == "__main__":
    # scan_variants = [111]

    plot_data = False
    scan_variants = [4, 5, 6, 7]

    thresholds = {
        "fit_improvement": [0.01, 0.03, 0.05],
        "err_reduction": [0.01, 0.1, 0.4, 0.8],
    }
    
    # Select whether to use mean or overall score for report
    score_aggregation = "mean"
    # score_aggregation = "total"

    for scan_variant in scan_variants:
        for score_measure, improvement_thresholds in thresholds.items():
            use_error_reduction = score_measure == "err_reduction"
            df_scores = {}
            outdir = config.PROJECT_DIR / "output" / f"synthetic_data_scan{scan_variant}"
            fitdir = config.FIT_DIR / "synthetic_data_fits" / f"scan{scan_variant}"
            for improvement_threshold in improvement_thresholds:
                df_scores[improvement_threshold] = run(outdir, fitdir, improvement_threshold, 
                                                    scan_variant, plot_data,
                                                    use_error_reduction, score_aggregation)
            plotScores(df_scores, scan_variant, use_error_reduction, score_aggregation, show=False)
    
    
    

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import datainput
import defaults
import config
import utils
import fitting
import plotting
from pprint import pp
from time import sleep
from plotting import plotExperiment, plotFitImprovement, plotFullFit, plotGrowthFit
from fitting import DEBFitter, fitFullModel, selectInteractions


def fitUptake(results_fn, multiSubstrateFitter, parallel=True):
    results = utils.loadState(results_fn)    
    uptakeParams, uptakeErrors  = fitting.fitUptakeParameterBatch({multiSubstrateFitter.getID() : multiSubstrateFitter}, 
                                                                  [multiSubstrateFitter.getID()],
                                                                  parallel = parallel)
    
    results.setdefault("uptake", {"params":{}, "errors":{}})
    results["uptake"]["params"].update(uptakeParams)
    results["uptake"]["errors"].update(uptakeErrors)
    
    pp(results)
    utils.saveState(results, results_fn)
    print("Saved uptake parameters to '%s'"%results_fn)


def prepareFitter(omit_runs, plot_mix_data=False, show_plot=False):
    # Load single substrate experiments (currently just to compile 'substrates' list)
    singleSubstrateExperiment = datainput.loadSingleSubstrateData()
    singleSubstrates = utils.orderSubstrates(singleSubstrateExperiment.keys(), None, config.substrateOrderKey)
    substrates = utils.orderSubstrates(singleSubstrates+["Ile"], None, config.substrateOrderKey)

    # Load mix timeline
    multiSubstrateExperiment = datainput.loadMultiSubstrateData("mix11", omit_runs)
    params = defaults.makeParams(substrates, len(multiSubstrateExperiment._runIDs))
    multiSubstrateFitter = fitting.DEBFitter(multiSubstrateExperiment, params)

    if plot_mix_data:
        plotExperiment(multiSubstrateExperiment, show=show_plot)

    return multiSubstrateFitter


def fitInteractions(results_fn, multiSubstrateFitter, n_interactions, fitID, figdir=None, 
                    cluster_similarity_threshold=0.075, parallel=False):
    # Number of interactions to consider for the fit.
    # For 1 interaction, this will add results for each substrate and each cluster 
    # For 2 interactions, this will add results for each substrate and each pair of 
    # clusters under the key tuple(cluster1_id, cluster2_id)
    assert(n_interactions in (1,2))
    tries, maxtries = 0, 10
    while tries < maxtries+1:
        try:
            results = utils.loadState(results_fn)
            break
        except EOFError as e:
            print(f"Seems that file '{results_fn}' is corrupted or unfinished. Waiting 1 sec...")
            tries += 1
            sleep(1)
        if tries == maxtries:
            # Restart fitting uptakes
            utils.saveState({}, results_fn)
            fitUptake(results_fn, multiSubstrateFitter, parallel=False)
            break
    
    # Fitted uptake parameters    
    uptakeParams = results["uptake"]["params"][multiSubstrateFitter.getID()]
    print("uptakeParams:")
    for k, v in uptakeParams.items():
        print(k,":",v)
    multiSubstrateFitter.setUptakeParams(uptakeParams)
    
    if "interactions" in results:
        print("Interaction fits so far:")
        for k, d in results["interactions"]['params'].items():
            if d: print(k)
    
    print("\nFitting interaction parameters for %d-cluster interactions."%n_interactions)
    substrates = multiSubstrateFitter.getSubstrates()
    interactionParams, interactionErrors = fitting.fitInteractions(multiSubstrateFitter, substrates, 
                                                                   nrInteractions=n_interactions, fitID=fitID,
                                                                   useInitialParams=True,
                                                                   figdir=figdir,
                                                                   cluster_similarity_threshold=cluster_similarity_threshold,
                                                                   parallel=parallel)
    print("interactionParams:")
    for k,v in interactionParams.items():
        print("  ",k,":",v)
    print("interactionErrors:")
    for k,v in interactionErrors.items():
        print("  ",k,":",v)
    
    # Fill results without presuming to set all substrates simultaneously
    results.setdefault("interactions", {"params":{}, "errors":{}})
    for sid, d in interactionParams.items():
        results["interactions"]["params"].setdefault(sid,{})
        for cid, v in d.items():
            results["interactions"]["params"][sid][cid] = v
    for sid, d in interactionErrors.items():
        results["interactions"]["errors"].setdefault(sid,{})
        for k,v in d.items():
            results["interactions"]["errors"][sid][k] = v
    utils.saveState(results, results_fn)
    print("Saved interaction parameters for %d-cluster interaction to '%s'!"%(n_interactions, results_fn))


def plotClusters(multiSubstrateFitter, omitted, show=False):    
    # Plot cluster timelines
    multiSubstrateFitter.buildSubstrateClusters(config.FIG_DIR/"substrateClusters.svg")
    # clusterColors = config.clusterColors[7]
    clusterColors = config.substrate_colors
    rids = [r for r in [f"F{i}" for i in range(1,5)] if r not in omitted]
    for rid in rids:
        plotting.plotCluster(multiSubstrateFitter, rid, omitted, separateFigs=False, show=show, colors=clusterColors)


def fitGrowthWithoutInteractions(results_fn, multiSubstrateFitter, 
                                 growth_fit_type, refit=False):
        assert(growth_fit_type == "growth without interactions")
        results = utils.loadState(results_fn)
        if growth_fit_type in results:
            print("Loaded results:")        
            pp(results[growth_fit_type])
            if not refit:
                return
        else:
            results.setdefault(growth_fit_type, {"params":{}, "errors":{}})

        fid = multiSubstrateFitter.getID()
        print("\nFitting growth parameters without interaction for %s ..."%fid)
        multiSubstrateFitter.deactivateInteractions()
        
        multiSubstrateFitter.setUptakeParams(results["uptake"]["params"][fid])
        growthParamsFitted, errors = fitting.fitGrowthParameters({fid:multiSubstrateFitter}, [fid], withInteractions=False)
        results[growth_fit_type]["params"]["%s single fit"%multiSubstrateFitter.getID()] = growthParamsFitted
        results[growth_fit_type]["errors"]["%s single fit"%multiSubstrateFitter.getID()] = errors
        utils.saveState(results, results_fn)
        print("Saved growth parameters fitted without interaction to '%s'"%results_fn)
    

def activateSignificantInteractions(multiSubstrateFitter, results, 
                                    qualifyingRelativeImprovement = 0.2, 
                                    qualifyingAbsoluteImprovement = 0.01):
    multiSubstrateFitter.resetInteractions()
    selectedInteractions, allInteractions = fitting.selectInteractions(qualifyingRelativeImprovement, qualifyingAbsoluteImprovement, results)
    multiSubstrateFitter.setInteractions(selectedInteractions)        
    print("Using interactions:")
    pp(multiSubstrateFitter._dynamics._a)
    multiSubstrateFitter.activateInteractions()
    return selectedInteractions


def fitGrowthWithInteractions(results_fn, multiSubstrateFitter, growth_fit_type, refit=False):
    # Fit growth parameters
    results = utils.loadState(results_fn)
    
    if growth_fit_type in results:
        print("Loaded results:")
        pp(results[growth_fit_type])
        if not refit:
            return
        
    if growth_fit_type not in results:
        # Initialize entry
        results[growth_fit_type] = {}
        results[growth_fit_type]["params"] = {}
        results[growth_fit_type]["errors"] = {}
        results[growth_fit_type]["interaction"] = {}

    # Ensure that clusters are computed
    multiSubstrateFitter.buildSubstrateClusters()
    
    # Relative improvement in fitting error that is taken to qualify 
    # an interaction to be considered in the fit
    selectedInteractions = activateSignificantInteractions(multiSubstrateFitter, results)

    for k, v in selectedInteractions.items():
        interactionCluster = v[0]["cluster"]
        interactionSubstrate = k
        print("  %s->%s"%(interactionCluster, interactionSubstrate))

    print(f"\nFitting growth parameters with interactions for {multiSubstrateFitter.getID()}")
    growthParamsFitted, errors = fitting.fitGrowthParameters({multiSubstrateFitter.getID():multiSubstrateFitter}, 
                                                            [multiSubstrateFitter.getID()], withInteractions=True)
    
    if "%s single fit"%multiSubstrateFitter.getID() not in results[growth_fit_type]["params"]:
        results[growth_fit_type]["params"]["%s single fit"%multiSubstrateFitter.getID()] = {}
        results[growth_fit_type]["errors"]["%s single fit"%multiSubstrateFitter.getID()] = {}
        results[growth_fit_type]["interaction"]["%s single fit"%multiSubstrateFitter.getID()] = {}
    results[growth_fit_type]["params"]["%s single fit"%multiSubstrateFitter.getID()] = growthParamsFitted
    results[growth_fit_type]["errors"]["%s single fit"%multiSubstrateFitter.getID()] = errors
    results[growth_fit_type]["interaction"]["%s single fit"%multiSubstrateFitter.getID()] = selectedInteractions
        
    print("\nselected Interactions:")
    pp(selectedInteractions)
    print("\n# Growth params:")
    print(growthParamsFitted)
    print("\n# Errors:")
    print(errors)
    utils.saveState(results, results_fn)
    print("Saved growth parameters fitted with interaction to '%s'"%results_fn)



def collectResamplingFitResults(resultdir:Path):
    omit_list = [
        ["F1"], ["F2"], ["F3"], ["F4"],
        ["F1","F2"], ["F1","F3"],
        ["F1","F4"], ["F2","F3"],
        ["F2","F4"], ["F3","F4"],
        ]
    results = {}
    for omit in omit_list:
        fn = resultdir / f"results_mix11_omit[{','.join(omit)}].pickle"
        if not fn.exists():
            raise Exception(f"File '{fn.name}' not found in result directory ('{str(resultdir)}'). Please run main_inhibition_fit with option --omit=[{','.join(omit)}] first")
        with open(fn, "rb") as f:
            results["".join(omit)] = pickle.load(f)    
    return results


def improvemenStatsReport(results_fn, multiSubstrateFitter, plot=True, show=False):
    results = utils.loadState(results_fn)
    resampling_results = collectResamplingFitResults(Path(results_fn).parent)
    resampling_results["all_runs"] = results
    dfs_best, dfs_all = {}, {}
    for k, results_k in resampling_results.items():
        dfs_best[k], dfs_all[k] = fitting.bestFitInformation(multiSubstrateFitter, results_k)

    # Determine best fits per interaction
    df_best, df_all = fitting.bestFitInformation(multiSubstrateFitter, results, resampling_results)
    
    ## Plot comparison table for R2 improvement by inclusion of interactions
    plotting.plotFitImprovement(df_best, df_all)

    # Plot matrix of improvements by including one interaction
    plotting.plotImprovementsMatrices(df_all, fitter=multiSubstrateFitter)

    # Plot glucose time series with/without inhibition
    plotting.plotGlucoseTimeseries(multiSubstrateFitter, results)

    # TODO: Plot full fit with Nag→glucose inhibiton
    plotting.fullFitTimeseries(multiSubstrateFitter, results)

    e1 = results["full, refit without interactions"]["errors"]
    e2 = results["full, refit with interactions"]["errors"]
    R21 = 1-e1["ssr"]/e1["sst"]
    R22 = 1-e2["ssr"]/e2["sst"]
    print("\nR2 finalPassWithoutInteraction:", R21)
    print("\nR2 finalPassWithInteraction:", R22)
    


def prepareFitterPars(fitter:DEBFitter, results_fn, fit_type:str, setInteractions, 
                      useIndividualUptakeFits,
                      useFullFittedInteractions=False, recompute=False):
    # Set Fitter parameters for plotting or as initial values for full fit
    results = utils.loadState(results_fn)

    # Use uptake parameters of individual substrate fits
    if useIndividualUptakeFits:
        upPars = list(results["uptake"]["params"].values())[0]
    else:
        # must be full fit
        if fit_type in results and not recompute:
            upPars = results[fit_type]["params"].get("%s single fit"%fitter.getID(), results[fit_type]["params"])
        else:
            upPars = results["uptake"]["params"][fitter.getID()]
    fitter.setUptakeParams(upPars)
    
    if fit_type in results and not recompute:
        growthParams = results[fit_type]["params"].get("%s single fit"%fitter.getID(), results[fit_type]["params"])
    elif fit_type in full_fit_types:
        # Per default use growth fit with interactions
        growthParams = results[growth_fit_types[1]]["params"]["%s single fit"%fitter.getID()]
    else:
        growthParams = None
    if growthParams is not None:
        fitter.setGrowthParams(growthParams)

    fitter.resetInteractions()
    if setInteractions:
        if useFullFittedInteractions:
            pars = results[fit_type]["params"].get("%s single fit"%fitter.getID(), results[fit_type]["params"])
            interactions = pars["interactions"]
        else:
            interactions, bestParams = selectInteractions(qualifyingRelativeImprovement=0.2,
                                            qualifyingAbsoluteImprovement=0.01, 
                                            results=results)
        fitter.setInteractions(interactions)
        fitter.activateInteractions()
    else:
        fitter.deactivateInteractions()


def runFullModelFit(multiSubstrateFitter, results_fn, fit_type, 
                    setInteractions, fitInteractions, recompute=False):
    results = utils.loadState(results_fn)
    if fit_type in results and not recompute:
        print("Loaded results:")
        pp(results[fit_type])
        return

    # Set initial parameter values for fit
    prepareFitterPars(fitter = multiSubstrateFitter, 
                        results_fn = results_fn, fit_type=fit_type,
                        setInteractions = setInteractions,
                        useIndividualUptakeFits = True,
                        useFullFittedInteractions = False)
    
    params, errors = fitFullModel(multiSubstrateFitter, fitInteractions)
    results[fit_type] = {}
    results[fit_type]["errors"] = errors
    results[fit_type]["params"] = params
    utils.saveState(params=results, fn=results_fn)

        
def addFullSystemPars_noRefit(multiSubstrateFitter:DEBFitter, results_fn):
    # Set initial parameter values for fit
    results = utils.loadState(results_fn)
    prepareFitterPars(fitter = multiSubstrateFitter, 
                        results_fn = results_fn, fit_type=full_fit_types[0],
                        setInteractions = True,
                        useIndividualUptakeFits = True,                    
                        useFullFittedInteractions = False)
    fitID = "%s single fit"%multiSubstrateFitter.getID()
    results.setdefault(full_fit_types[0], {fitID:{}})
    results[full_fit_types[0]]["params"] = multiSubstrateFitter.getGrowthParams()
    results[full_fit_types[0]]["params"].update(multiSubstrateFitter.getUptakeParams())
    results[full_fit_types[0]]["params"]["V0"] = multiSubstrateFitter.getV0()
    results[full_fit_types[0]]["params"]["interactions"] = multiSubstrateFitter.getInteractions()
    ssr, sst = 0, 0
    for rid in multiSubstrateFitter.getRunIDs():
        ssri, ssti = multiSubstrateFitter.residualErrorFull(rid=rid, plot=False)
        ssr += ssri
        sst += ssti
    print("ssr:")
    pp(ssr)
    print("sst:")
    pp(sst)
    r2 = 1 - ssr/sst
    print(f"r2: {r2:g}")
    results[full_fit_types[0]]["errors"] = dict(ssr=ssr, sst=sst, r2=r2)
    utils.saveState(results, fn=results_fn)


def assess_improvements(results_fn, clusters):
    # First, load R2-values for all relevant fit results
    results = utils.loadState(results_fn)
    R2s = dict(
        uptake_no_int = {},
        uptake_1_int = {},
        uptake_2_int = {},
        full_no_refit = {},
        full_no_ints = {},
        full_ints = {},
        )
    substrates = sorted(results["uptake"]["errors"]["mix11"].keys())
    print("\nUsing substrate clusters:")
    pp(clusters)

    # Base uptake errors
    errs_no_int = results["uptake"]["errors"]["mix11"]
    pars_no_int = results["uptake"]["params"]["mix11"]
    for s, errs in errs_no_int.items():
        r2 = 1 - errs["ssr"]/errs["sst"]
        R2s["uptake_no_int"][s] = r2
        print(f"\nBase R² for '{s}': {r2:.04f}")
    
    # Improvements with one interaction
    errs_1_int = results["interactions"]["errors"]
    pars_1_int = results["interactions"]["params"]
    for s1 in substrates:
        # pp(errs_1_int[s1])
        R2s["uptake_1_int"][s1] = {}
        dR2_max = -np.inf # maximal absolute improvement
        r20 = R2s["uptake_no_int"][s1]
        notable_fits = set() # other sub-optimal fits for s1 with dR2 > th
        th_notable = 0.025
        best_fit = None
        significance_str = ""
        for s2 in clusters:
            if s1 in clusters[s2]:
                continue # no fit to cluster containing s1
            # fit for inhibition s2 → s1
            errs = errs_1_int[s1][(s2,)]
            pars = pars_1_int[s1][(s2,)]
            r2 = 1 - errs["ssr"]/errs["sst"]
            dR2 = r2-r20
            r = (r2-r20)/r20
            q = (r2-r20)/(1-r20)
            R2s["uptake_1_int"][s1][(s2,)] = {
                "R2": r2, 
                "dR2": dR2, 
                "rel. error reduction": q, 
                "rel. fit improvement": r,
                "a": pars["a"],
                "mu": pars["mu"],
                "K": pars["K"],
            }
            if dR2 > dR2_max:
                best_fit = (s2,)
                dR2_max = dR2
                if dR2 > th_notable:
                   significance_str = "(*) " 
            if dR2 > th_notable:
                notable_fits.add((s2,))
        R2s["uptake_1_int"][s1]["best"] = best_fit
        print(f"\n{significance_str}Best interaction fit for '{s1}' is '{best_fit}':")
        print(f"Base R² (no interactions): {r20}")
        pp(R2s["uptake_1_int"][s1][best_fit])
        
        notable_fits = sorted(notable_fits.difference([best_fit]))
        R2s["uptake_1_int"]["notable"] = notable_fits
        for s2 in notable_fits:
            print(f"\n(*) Additional good fit for '{s1}' is '{s2}':")
            print(f"Base R² (no interactions): {r20}")
            pp(R2s["uptake_1_int"][s1][s2])


    # Improvements with two interaction
    errs_2_int = results["interactions"]["errors"]
    for s1 in substrates:
        R2s["uptake_2_int"][s1] = {}
        dR2_max = -np.inf # maximal absolute improvement
        r20 = R2s["uptake_1_int"][s1][R2s["uptake_1_int"][s1]["best"]]["R2"]
        notable_fits = set() # other sub-optimal fits for s1 with dR2 > th
        th_notable = 0.025
        best_fit = None
        significance_str = ""
        for s2 in clusters:        
            for s3 in clusters:
                if s1 in clusters[s2] or s1 in clusters[s3] or s2==s3:
                    continue # no fit to cluster containing s1
                errs = errs_1_int[s1][(s2, s3)]
                r2 = 1 - errs["ssr"]/errs["sst"]
                dR2 = r2-r20
                r = (r2-r20)/r20
                q = (r2-r20)/(1-r20)
                R2s["uptake_2_int"][s1][(s2,s3)] = {
                    "R2": r2, 
                    "dR2": dR2, 
                    "rel. error reduction": q, 
                    "rel. fit improvement": r,
                }
                if dR2 > dR2_max:
                    best_fit = (s2, s3)
                    dR2_max = dR2
                    if dR2 > th_notable:
                       significance_str = "(*) " 
                if dR2 > th_notable:
                    notable_fits.add((s2,s3))

        R2s["uptake_2_int"][s1]["best"] = best_fit
        print(f"\n{significance_str}Best 2-interaction fit for '{s1}' is '{best_fit}':")
        print(f"Base R² (best 1-interactions fit): {r20}")
        pp(R2s["uptake_2_int"][s1][best_fit])
        
        notable_fits = sorted(notable_fits.difference([best_fit]))
        R2s["uptake_2_int"]["notable"] = notable_fits
        for s2s3 in notable_fits:
            print(f"\n(*) Additional good fit for '{s1}' is '{s2s3}':")
            print(f"Base R² (no interactions): {r20}")
            pp(R2s["uptake_2_int"][s1][s2s3])

    
    # Keys for DataFrame of improvements
    # "substrate", "R2", "nrInteractions", "rel. error reduction"
    df = {"substrate":[], "R2":[], "nrInteractions":[], "dR2":[],
          "rel. error reduction":[], "rel. fit improvement":[],
          "interactions":[]}
    for s in substrates:
        df["substrate"].extend([s]*3)
        df["nrInteractions"].extend([0,1,2])
        best_1int = R2s["uptake_1_int"][s][R2s["uptake_1_int"][s]["best"]]
        best_2int = R2s["uptake_2_int"][s][R2s["uptake_2_int"][s]["best"]]
        df["R2"].append(R2s["uptake_no_int"][s])
        df["R2"].append(best_1int["R2"])
        df["R2"].append(best_2int["R2"])
        df["interactions"].append(tuple())
        df["interactions"].append(R2s["uptake_1_int"][s]["best"])
        df["interactions"].append(R2s["uptake_2_int"][s]["best"])
        for k in ["dR2", "rel. error reduction", "rel. fit improvement"]:
            df[k].append(0.0)
            df[k].append(best_1int[k])
            df[k].append(best_2int[k])
    df = pd.DataFrame(df)

    # Assessment improvements full fit 
    errs_full_no_refit = results[full_fit_types[0]]["errors"]
    errs_full_no_ints = results[full_fit_types[1]]["errors"]
    errs_full_ints = results[full_fit_types[2]]["errors"]
    R2s["full_no_refit"] = errs_full_no_refit
    R2s["full_no_ints"] = errs_full_no_ints
    R2s["full_ints"] = errs_full_ints
    print("\nFull fit R2s:")
    for k in ["full_no_refit", "full_no_ints", "full_ints"]:
        r2 = 1 - R2s[k]["ssr"]/R2s[k]["sst"]
        R2s[k]["r2"] = r2
        print(f"{k}: {r2:.04f}")
    # assess improvement without → with interactions
    r2, r20 = R2s["full_ints"]["r2"], R2s["full_no_ints"]["r2"]
    dR2 = r2-r20
    r = (r2-r20)/r20
    q = (r2-r20)/(1-r20)
    R2s["full_ints"]["dR2"] = dR2
    R2s["full_ints"]["rel. fit improvement"] = r
    R2s["full_ints"]["rel. error reduction"] = q
    print("\nImprovements full fit: without → with interactions:")
    print(f"  dR2 = {dR2:.04f}")
    print(f"  rel. fit improvement = {r:.04f}")
    print(f"  rel. error reduction = {q:.04f}")


def main(results_fn, omit_runs):
    # Flags for turning parts of the script on/off
    fit_uptake=True
    fit_1int=True
    fit_2int=True
    growth_fit=True
    full_fit=True

    report_improvement_stats = len(omit_runs)==0 # i.e., for full fit
    assessment = len(omit_runs)==0 # i.e., for full fit

    plot_mix_data=True and (len(omit_runs)==0)
    multiSubstrateFitter = prepareFitter(plot_mix_data=plot_mix_data, omit_runs=omit_runs, show_plot=False)
    
    if len(omit_runs)==0:
        plotClusters(multiSubstrateFitter, omitted=omit_runs, show=False)
    
    if not results_fn.exists():
        utils.saveState({}, results_fn)
    else:
        results = utils.loadState(results_fn)
        print(f"Results file '{results_fn}' exist\nContents:")
        pp(results)

    # Fit uptake parameters seperately for all compounds
    if fit_uptake:
        fitUptake(results_fn, multiSubstrateFitter)
    
    # Fit for one interaction
    fitID = "interaction_fit_all_runs" if len(omit_runs) == 0 else f"interaction_fit_omit{''.join(omit_runs)}"
    if fit_1int:
        fitInteractions(results_fn, multiSubstrateFitter, n_interactions=1, fitID=fitID, parallel=True)
    # Fit for two interactions
    if fit_2int:
        fitInteractions(results_fn, multiSubstrateFitter, n_interactions=2, fitID=fitID, parallel=True)
    
    if growth_fit:
        # Fit with and without interactions
        for growth_fit_type in growth_fit_types:
            if growth_fit_type == growth_fit_types[0]:
                print("\nFitting growth subsystem without interactions")
                fitGrowthWithoutInteractions(results_fn, multiSubstrateFitter, 
                                            growth_fit_type=growth_fit_type,
                                            refit=False)
            else:
                print("\nFitting growth subsystem with interactions")
                fitGrowthWithInteractions(results_fn, multiSubstrateFitter, 
                                            growth_fit_type=growth_fit_type,
                                            refit=False)
            # Plot
            # Set Fitter parameters for plotting
            results = utils.loadState(results_fn)
            prepareFitterPars(fitter=multiSubstrateFitter, 
                                results_fn=results_fn, fit_type=growth_fit_type,
                                setInteractions=growth_fit_type==growth_fit_types[1], 
                                useIndividualUptakeFits=True,
                                recompute=False)
            growthParams = results[growth_fit_type]["params"]["%s single fit"%multiSubstrateFitter.getID()]
            errors = results[growth_fit_type]["errors"]["%s single fit"%multiSubstrateFitter.getID()]
            if len(omit_runs) == 0:
                plotGrowthFit(multiSubstrateFitter, growthParams, 
                            errors, multiSubstrateFitter.getRunIDs(), 
                            multiSubstrateFitter.getID() + f" growth-subsystem ({growth_fit_type})", 
                            show=True)
    
    if full_fit:
        ## Test full model fit in various variants
        # No refitting 
        addFullSystemPars_noRefit(multiSubstrateFitter, results_fn)
        prepareFitterPars(fitter = multiSubstrateFitter, 
                            results_fn = results_fn, fit_type = full_fit_types[0],
                            setInteractions = True,
                            useIndividualUptakeFits = True,
                            useFullFittedInteractions = False)
        if len(omit_runs) == 0:
            plotFullFit(results_fn, multiSubstrateFitter, full_fit_type=full_fit_types[0])
        # Fit without interactions
        runFullModelFit(multiSubstrateFitter, results_fn,
                        setInteractions=False,
                        fitInteractions=False,
                        fit_type=full_fit_types[1],
                        recompute = False)
        prepareFitterPars(fitter = multiSubstrateFitter, 
                            results_fn = results_fn, fit_type = full_fit_types[1],
                            setInteractions = False,
                            useIndividualUptakeFits = True,
                            useFullFittedInteractions = False)
        if len(omit_runs) == 0:
            plotFullFit(results_fn, multiSubstrateFitter, full_fit_type=full_fit_types[1])
        # Fit with interactions
        runFullModelFit(multiSubstrateFitter, results_fn,
                        setInteractions=True,
                        fitInteractions=True,
                        fit_type=full_fit_types[2],
                        recompute = False)
        prepareFitterPars(fitter = multiSubstrateFitter, 
                            results_fn = results_fn, fit_type = full_fit_types[2],
                            setInteractions = True,
                            useIndividualUptakeFits = False,
                            useFullFittedInteractions = True)
        if len(omit_runs) == 0:
            plotFullFit(results_fn, multiSubstrateFitter, full_fit_type=full_fit_types[2])
        
    if report_improvement_stats:
        improvemenStatsReport(results_fn, multiSubstrateFitter, show=True)

    if assessment:
        multiSubstrateFitter.buildSubstrateClusters()
        clusters:dict = multiSubstrateFitter.getClusters()
        substrates = multiSubstrateFitter._substrates
        clusters = {cid : sorted([substrates[i] for i in clusters[cid]]) for cid in sorted(clusters)}
        assess_improvements(results_fn, clusters)


def parse_args():
    parser = ArgumentParser(prog='Subsmix fitter',
                    description='Fit inhibitory interactions for substrate mixture growth.',)
    parser.add_argument("--omit", help="Omit specific runs for resampling fits. (Subset of F1, F2, F3, F4)", type=str, default="")
    args = parser.parse_args()
    if args.omit:
        args.omit = sorted(args.omit.split(","))
        for x in args.omit:
            assert(x in [f"F{i}" for i in range(1,5)])
    else:
        args.omit=[]
    return args



if __name__ == '__main__':

    full_fit_types = [
        "full, no refit with interactions",
        "full, refit without interactions",
        "full, refit with interactions"
    ]

    growth_fit_types = [
        "growth without interactions",
        "growth with interactions"
    ]

    args = parse_args()    
    if args.omit:
        omit_str = ",".join(args.omit)
        omit_str = f"_omit[{omit_str}]"
    else:
        omit_str = "_all_runs"
    results_fn = config.FIT_DIR / f"results_mix11{omit_str}.pickle"

    main(results_fn, omit_runs=args.omit)
    
    

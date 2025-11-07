import os
from pathlib import Path
import re
import multiprocessing as mp
from matplotlib.pyplot import sca
import numpy as np
from pprint import pp

from config import PROJECT_DIR, NCPU
from data_synthesizer import generate_synthetic_experiment, save_experiment_data
from main_P_inhibens_fit import prepareFitter, fitUptake, fitInteractions,\
    fitGrowthWithoutInteractions, fitGrowthWithInteractions
import config
import utils
import datainput
import fitting
import sys
from defaults import default_for_sampled_params
from itertools import starmap
from numpy.random._generator import default_rng

def get_scan_variant_description(scan_variant):
    if scan_variant == 1:
        return dict(
            title="Joint variation of noise and system size",
            steps=[0.0, 0.1, 1.0, 2.0, 3.0, 6.0, 10.0],
            ticks=4 + np.round(np.array([0.0, 0.1, 1.0, 2.0, 3.0, 6.0, 10.0])*2),
            label="#subs",
            )
    elif scan_variant == 2:
        return dict(
            title="Only noise variation for nsubs=nint=15",
            steps=[0.0, 0.1, 1.0, 2.0, 3.0, 6.0, 10.0],
            ticks=np.array([0.0, 0.1, 1.0, 2.0, 3.0, 6.0, 10.0])*0.1,
            label="noise",
            )
    elif scan_variant == 3:
        return dict(
            title="Only size variation without noise",
            steps=[0.0, 0.1, 1.0, 2.0, 3.0, 6.0, 10.0],
            ticks=4 + np.round(np.array([0.0, 0.1, 1.0, 2.0, 3.0, 6.0, 10.0])*1.3),
            label="#subs",
            )
    elif scan_variant == 4:
        return dict(
            title="# Only dynamic noise (#subs=6, #int=4)",
            steps=np.array([0.0, 1.0, 2.0, 3.0, 6.0, 10.0, 15.0, 20.0]),
            ticks=np.array([0.0, 1.0, 2.0, 3.0, 6.0, 10.0, 15.0, 20.0])*0.1,
            label="noise intensity",
            )
    elif scan_variant == 5:
        return dict(
            title="# Only observation noise (#subs=6, #int=4)",
            steps=np.array([0.0, 1.0, 2.0, 3.0, 6.0, 10.0, 15.0, 20.0]),
            ticks=np.array([0.0, 1.0, 2.0, 3.0, 6.0, 10.0, 15.0, 20.0])*0.1,
            label="noise intensity",
            )
    elif scan_variant == 6:
        return dict(
            title="# Only interaction number",
            steps=np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 16.0, 20.0]),
            ticks=np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 16.0, 20.0]),
            label="# interactions",
            )
    elif scan_variant == 7:
        return dict(
            title="# System size and interaction number",
            steps=np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]),
            ticks=4 + np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]),
            label="# substrates",
            )
    elif scan_variant == 111:
        return dict(
            title="# Dynamic noise illustration (#subs=6, #int=4)",
            steps=np.array([0.0, 1.0, 2.0, 3.0, 6.0, 10.0, 15.0, 20.0]),
            ticks=np.array([0.0, 1.0, 2.0, 3.0, 6.0, 10.0, 15.0, 20.0])*0.1,
            label="noise intensity",
            )
    else:
        raise Exception(f"No scan variant '{scan_variant}' implemented.")


def get_scan_variant(scan_variant):
    # Scan variants determine an affine slice 
    # for the system characteristics to be probed,
    # as defined by base and scale values.
    # Used param combinations are calculated as
    #   base + scale*factor 
    
    if scan_variant == 1:
        # Joint variation of noise and system size
        master_seed = 9876
        scales = dict(
            nsubs = 2, nint  = 3,
            I_dyn = 0.1, I_obs = 0.1,
            )
        bases = dict(
            nsubs = 4, nint  = 1.5,
            I_dyn = 0.0, I_obs = 0.0,
            )
        factors = np.array([0.0, 0.1, 1.0, 2.0, 3.0, 6.0, 10.0])
        fit2interactions = True
    elif scan_variant == 2:
        # Only noise variation for nsubs=nint=15
        master_seed = 9877
        scales = dict(
            nsubs = 0, nint  = 0,
            I_dyn = 0.1, I_obs = 0.1,
            )
        bases = dict(
            nsubs = 15, nint  = 15,
            I_dyn = 0.0, I_obs = 0.0,
            )
        factors = np.array([0.0, 0.1, 1.0, 2.0, 3.0, 6.0, 10.0])
        fit2interactions = True
    elif scan_variant == 3:
        # Only size variation without noise
        master_seed = 9879
        scales = dict(
            nsubs = 1.3, nint  = 2,
            I_dyn = 0.0, I_obs = 0.0,
            )
        bases = dict(
            nsubs = 4, nint  = 2,
            I_dyn = 0.0, I_obs = 0.0,
            )
        factors = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 10.0])
        fit2interactions = True
    elif scan_variant == 4:
        # Only dynamic noise
        master_seed = 5432
        scales = dict(
            nsubs = 0, nint  = 0,
            I_dyn = 0.1, I_obs = 0.0,
            )
        bases = dict(
            nsubs = 6, nint  = 4,
            I_dyn = 0.0, I_obs = 0.0,
            )
        factors = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 10.0, 15.0, 20.0])
        par_count_offset = 0
        fit2interactions = True
    elif scan_variant == 5:
        # Only observation noise
        master_seed = 5433
        scales = dict(
            nsubs = 0, nint  = 0,
            I_dyn = 0.0, I_obs = 0.1,
            )
        bases = dict(
            nsubs = 6, nint  = 4,
            I_dyn = 0.0, I_obs = 0.0,
            )
        factors = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 10.0, 15.0, 20.0])
        par_count_offset = 0
        fit2interactions = True
    elif scan_variant == 6:
        # Only interaction nr
        master_seed = 5433
        scales = dict(
            nsubs = 0, nint  = 1,
            I_dyn = 0.0, I_obs = 0.0,
            )
        bases = dict(
            nsubs = 6, nint  = 0,
            I_dyn = 0.0, I_obs = 0.0,
            )
        factors = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 16.0, 20.0])
        par_count_offset = 0
        fit2interactions = True
    elif scan_variant == 7:
        # Only interaction nr and system size
        master_seed = 5433
        scales = dict(
            nsubs = 1, nint  = 1,
            I_dyn = 0.0, I_obs = 0.0,
            )
        bases = dict(
            nsubs = 4, nint  = 2,
            I_dyn = 0.0, I_obs = 0.0,
            )
        factors = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        par_count_offset = 0
        fit2interactions = False
    elif scan_variant == 111:
        # Dynamic noise illustration
        master_seed = 55555
        scales = dict(
            nsubs = 0, nint  = 0,
            I_dyn = 0.1, I_obs = 0.0,
            )
        bases = dict(
            nsubs = 6, nint  = 4,
            I_dyn = 0.0, I_obs = 0.0,
            )
        # par_count_offset = 0
        factors = np.array([1.0, 20.0])
        par_count_offset = 1
        fit2interactions = False
    else:
        raise Exception(f"No scan variant '{scan_variant}' implemented.")
    
    outdir = PROJECT_DIR / "output" / f"synthetic_data_scan{scan_variant}"
    fitdir = config.FIT_DIR / "synthetic_data_fits" / f"scan{scan_variant}"
    
    if not outdir.exists():
        os.makedirs(outdir, exist_ok=True)
    if not fitdir.exists():
        os.makedirs(fitdir, exist_ok=True)

    return dict(
        scan_variant=scan_variant,
        master_seed=master_seed,
        scales=scales,
        bases=bases,
        factors=factors,
        outdir=outdir,
        fitdir=fitdir,
        par_count_offset=par_count_offset,
        fit_ints2=fit2interactions,
        )


def generate_synthetic_data(master_seed, params, runs_per_combi, par_count_offset, outdir,
                            scan_variant, plot_trajs=False, rate_plot=False, plot_nw=False):
    
    if not outdir.exists():
        os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(master_seed)
    
    # Time points for the observed data
    tend = 150
    tspan_obs = np.arange(0,tend,4)    
    # experiment duplicates
    nruns = 4

    fixed_seeding = scan_variant == 111
    if fixed_seeding:
        # To illustrate parametric influence isolated from seed
        combi_seed = rng.integers(0, 99999999)

    for par_count, pars in enumerate(params):
        if not fixed_seeding:
            combi_seed = rng.integers(0, 99999999)
        combi_rng = default_rng(combi_seed)
        nsubs = pars["nsubs"]
        total_interaction_nr = pars["nint"]
        dynamic_noise_intensity = pars["I_dyn"]
        measurement_noise_intensity = pars["I_obs"]
        for run_count in range(runs_per_combi):
            seed = combi_rng.integers(0, 99999999)
            # Experiment title
            expID = f"synth_exp_{par_count+par_count_offset}-{run_count}_seed{seed}"
            exp = generate_synthetic_experiment(outdir=outdir,
            nruns=nruns, tend=tend, tspan_obs=tspan_obs, 
            nsubs=nsubs, total_interaction_nr=total_interaction_nr,
            seed=seed, dynamic_noise_intensity=dynamic_noise_intensity, 
            measurement_noise_intensity=measurement_noise_intensity,
            plot_trajs=plot_trajs, plot_nw=plot_nw, save_params=True,
            rate_plot=rate_plot,
            expID=expID
            )
            save_experiment_data(exp, outdir)
    
    


def make_params(scales, bases, factors):
    params = []
    for f in factors:
        pars_f = {}
        for k in sorted(bases):
            pars_f[k] = bases[k] + f*scales[k]
            if k in ["nsubs", "nint"]:
                pars_f[k] = int(np.round(pars_f[k]))
            else:
                pars_f[k] = np.round(pars_f[k], 9)
        params.append(pars_f)
    return params
    
    
def prepareFitter(data_fn, experimentID):
    # Load mix timeline
    multiSubstrateExperiment = datainput.loadSyntheticData(data_fn, experimentID)
    runIDs = sorted(multiSubstrateExperiment._tSpan.keys())
    substrates = sorted(multiSubstrateExperiment._sSpan[runIDs[0]].keys())
    
    params = default_for_sampled_params(len(runIDs), substrates)
    multiSubstrateFitter = fitting.DEBFitter(multiSubstrateExperiment, params)
    multiSubstrateFitter._synthetic = True
    return multiSubstrateFitter


def fit_synthetic_data(data_fn, fitdir, pars, parallel):
    
    fit_uptake = pars.get("fitUptake", True)
    fit_ints1 = pars.get("fit_ints1", True)
    fit_ints2 = pars.get("fit_ints2", True)
    fit_growth_without = pars.get("fit_growth_without", False)
    fit_growth_with = pars.get("fit_growth_with", False)
                       
    p = re.compile(r"[a-z_]+[0-9-]+_seed([0-9]+)\.[a-z]+")
    seed = re.match(p, data_fn.name).group(1)
    
    multiSubstrateFitter = prepareFitter(data_fn, experimentID=f"exp{seed}")
    
    results_fn = fitdir / f"results_synthetic_data_seed{seed}.pickle"
    if not results_fn.exists():
        utils.saveState({}, results_fn)
        print(f"Initialized result file '{results_fn}'")
        
    fitID = Path(data_fn).parent.name

    if fit_uptake:
        # # Fit uptake parameters seperately for all compounds
        fitUptake(results_fn, multiSubstrateFitter, parallel=parallel)

    if fit_ints1:
        # # Fit for one interaction
        fitInteractions(results_fn, multiSubstrateFitter, 
                        n_interactions=1, fitID=fitID, figdir=fitdir,
                        cluster_similarity_threshold=0.075, parallel=parallel)
    
    if fit_ints2:
        # # Fit for two interactions
        fitInteractions(results_fn, multiSubstrateFitter, 
                        n_interactions=2, fitID=fitID,
                        cluster_similarity_threshold=0.075, parallel=parallel)

    # # Fit growth parameters
    if fit_growth_without:
        fitGrowthWithoutInteractions(results_fn, multiSubstrateFitter, growth_fit_type="growth without interactions")
    if fit_growth_with:
        fitGrowthWithInteractions(results_fn, multiSubstrateFitter, growth_fit_type="growth with interactions")

    return results_fn
    

def run(pars, master_seed, runs_per_combi):
    # Explore dependence of parameter reconstruction on noise and system size.
    # Varying one or several of the following characteristics:
    #    - number of substrates (mixture complexity)
    #    - number of interactions (regulation complexity)
    #    - magnitude of dynamic parameter noise
    #    - magnitude of observation noise    
    scales, bases, factors = pars["scales"], pars["bases"], pars["factors"]
    generate_data = pars["generate_data"]
    par_count_offset = pars.get("par_count_offset", 0)
    params = make_params(scales, bases, factors)
    
    print("\nparams:")
    pp(params)
    
    # Multiprocessing fits: Either all different experiments (scan points) are launched in parallel,
    # or each fit uses internal parallelization (daemonic processes cannot have children processes on their own...)
    parallel_scan_points = True
    parallel_fitting_per_run = not parallel_scan_points    
    
    outdir = pars["outdir"]
    if master_seed is None:
        master_seed = pars["master_seed"]
    
    if generate_data:
        plot = pars["scan_variant"] == 111
        # Set plot_nw == True to generate Fig. S7(a)
        generate_synthetic_data(master_seed, params, runs_per_combi, par_count_offset, 
                                outdir, scan_variant=pars["scan_variant"], plot_trajs=plot, rate_plot=plot, plot_nw=False)
    
    if pars["scan_variant"] == 111:
        return

    param_fns = [fn for fn in outdir.iterdir() if fn.name.split(".")[-1]=="toml"]
    data_fns = [fn for fn in outdir.iterdir() if fn.name.split(".")[-1]=="csv"]
    
    p = re.compile(r"[a-z_]+[0-9-]+_seed([0-9]+)\.[a-z]+")
    param_fns = {re.match(p, fn.name).group(1):fn for fn in param_fns}
    data_fns = {re.match(p, fn.name).group(1):fn for fn in data_fns}
    assert(set(param_fns) == set(data_fns))
    
    seeds = sorted(param_fns)
    fitdir = pars["fitdir"]
    
    # Run fits
    args = list(reversed([(data_fns[seed], fitdir, pars, parallel_fitting_per_run) for seed in seeds]))
    if parallel_scan_points:
        pool = mp.Pool(NCPU)
        results = pool.starmap(fit_synthetic_data, args)
    else:
        results = starmap(fit_synthetic_data, args)
    results = [res for res in results]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        assert(len(sys.argv) == 4)
        scan_variant = int(sys.argv[1])
        master_seed = int(sys.argv[2])
        runs_per_combi = int(sys.argv[3])
    else:
        # Number of runs per factor, i.e. param combi)
        runs_per_combi = 1
        scan_variant = 111
        master_seed = 2216
    print("\nmain_fit_synthetic_data.py")
    print("run parameter:")
    print(f"scan_variant: {scan_variant}")
    print(f"master_seed: {master_seed}")
    print(f"runs_per_combi: {runs_per_combi}")
    pars = get_scan_variant(scan_variant)
    pars["generate_data"] = True
    print(f"scan pars:")
    pp(pars)
    run(pars, master_seed, runs_per_combi)
    
    
    

from pathlib import Path
import re
import os
import sys
import pandas as pd
import numpy as np
import pickle
from pprint import pp

import config
import batchexperiment
from config import DATA_DIR, runLabels, EXP_FNS, MIX_SUBDIRS,\
    SYNTHETIC_DATA_DIR


def loadSingleSubstrateData():
    experiments = {}
    datasubdir = DATA_DIR / MIX_SUBDIRS["pure"] 
    for run in runLabels:
        pat = re.compile(EXP_FNS["pure"][run].replace("*", "(.+)"))
        fns = dict([(re.match(pat,fn.name).group(1), fn) for fn in datasubdir.iterdir() if re.match(pat, fn.name)])
        for (k,fn) in fns.items():
            print("Loading experimental data from '%s'"%str(fn))
            try:
                df = pd.read_csv(fn)
            except pd.errors.EmptyDataError:
                print("File not present for run '%s' of '%s'!"%(run,k))
                continue
            data = {k: np.array(df[k]) for k in df.columns}
            # Experiment class holds S as dict: subs → timeseries
            data["S"] = {k:data["S"]}            
            if not k in experiments:
                experiments[k] = batchexperiment.BatchExperiment(k)            
            experiments[k].addRun(run, data)
    return experiments


def loadSyntheticData(data_fn, experimentID):
    experiment = batchexperiment.BatchExperiment(experimentID)
    df = pd.read_csv(data_fn)
    substrates = sorted([c for c in df.columns if c[0]=="S"])
    gg = df.groupby("rid").groups
    runIDs = sorted(gg)
    for rid in runIDs:
        ix = gg[rid]
        dfr = df.iloc[ix,:]
        data= dict(
            t = np.array(dfr["t"]),
            CDW = np.array(dfr["CDW"]),
            CDWOD = np.array(dfr["CDWOD"]),
            S = {s:np.array(dfr[s]) for s in substrates}
            )
        experiment.addRun(rid, data)
    return experiment


def loadMultiSubstrateData(experimentID, omit_runs): 
    experiment = batchexperiment.BatchExperiment(experimentID)
    for rid in runLabels:
        if rid in omit_runs:
            print(f"Omitting run '{rid}'")
            continue 
        ## Load growth data
        in_fn = os.path.join(DATA_DIR, MIX_SUBDIRS[experimentID], EXP_FNS[experimentID][rid])
        print("Loading",in_fn)
        df = pd.read_csv(in_fn)
        data = {k: np.array(df[k]) for k in df.columns}
        data["S"] = {}
        for s in  config.subs_names:
            data["S"][s] = np.array(df[s])
        experiment.addRun(rid, data)
    return experiment


def load_synthetic_data(data_id):
    fn = SYNTHETIC_DATA_DIR / f"{data_id}.pickle"
    with open(fn, "rb") as f:
        data = pickle.load(f)
    print("Loaded synthesized data from {fn}.")
    print("data:")
    pp(data)
    
    experimentID = data["experimentID"]
    params = data["params"]
    experiment = batchexperiment.BatchExperiment(experimentID)
    
    run_ids = sorted(data["runs"].keys())
    for rid in run_ids:
        run = data["runs"][rid]
        Nt = len(run["t"])
        cpd_names = sorted(run["S"].keys())
        experiment.addRun(rid, run)
    
    
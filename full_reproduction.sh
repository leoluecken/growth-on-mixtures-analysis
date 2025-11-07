#!/bin/bash

# Preconditions:
#  - SUBSMIX_HOME is defined (pointing to cloned git directory)
#  - a virtualenv environment has been created in $SUBSMIX_HOME/.venv

# Change to project root directory
cd "$SUBSMIX_HOME"

# Add scripts to PYTHONPATH
export PYTHONPATH=$SUBSMIX_HOME/src:$PYTHONPATH

# activate python env
source .venv/bin/activate

# Run fits for P. inhibens experiments on reduced replicate sets for resampling. (Can be run in parallel.)
python $SUBSMIX_HOME/src/main_P_inhibens_fit.py --omit=F1
python $SUBSMIX_HOME/src/main_P_inhibens_fit.py --omit=F2
python $SUBSMIX_HOME/src/main_P_inhibens_fit.py --omit=F3
python $SUBSMIX_HOME/src/main_P_inhibens_fit.py --omit=F4
python $SUBSMIX_HOME/src/main_P_inhibens_fit.py --omit=F1,F2
python $SUBSMIX_HOME/src/main_P_inhibens_fit.py --omit=F1,F3
python $SUBSMIX_HOME/src/main_P_inhibens_fit.py --omit=F1,F4
python $SUBSMIX_HOME/src/main_P_inhibens_fit.py --omit=F2,F3
python $SUBSMIX_HOME/src/main_P_inhibens_fit.py --omit=F2,F4
python $SUBSMIX_HOME/src/main_P_inhibens_fit.py --omit=F3,F4

# Run full fit for P. inhibens experiments (first complete resampling runs!) and assess results
python $SUBSMIX_HOME/src/main_P_inhibens_fit.py


# Generate synthetic data . (Can be run in parallel.)
# (options for main_fit_synthetic_data.py: 1: scan_variant; 2: master_seed; 3: runs_per_combi)
# Dynamic noise illustration
python $SUBSMIX_HOME/src/main_fit_synthetic_data.py 111 2216 1
# Dynamic noise scan
python $SUBSMIX_HOME/src/main_fit_synthetic_data.py 4 2211 14
# Observation noise scan
python $SUBSMIX_HOME/src/main_fit_synthetic_data.py 5 2212 14
# Interaction number scan
python $SUBSMIX_HOME/src/main_fit_synthetic_data.py 6 2213 14
# Interaction number and system size scan
python $SUBSMIX_HOME/src/main_fit_synthetic_data.py 7 2114 14


# Analyse synthetic data fits (run fit batch first!)
python $SUBSMIX_HOME/src/main_assess_synthetic_data_fit.py

# Fit create example figures for model illustration (Fig S1)
python $SUBSMIX_HOME/src/main_model_illustration.py

# Fit growth on single substrates and compare to mixture growth (run main_P_inhibens_fit.py runs first!)
python $SUBSMIX_HOME/src/main_single_substrates.py






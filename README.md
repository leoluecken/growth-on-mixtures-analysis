# Preparation

This assumes you are working on a Linux-System and have set the environment variable 'SUBSMIX_HOME' to the directory, where the software should be located, e.g.

    export SUBSMIX_HOME="$HOME/repos/subsmix"

The Python version used to generate the results in the paper is 3.11.3

1) Install virtualenv and create a virtual environment in the project base directory:
```
    cd $SUBSMIX_HOME
    virtualenv .venv
```

2) Activate environment and install required Python packages (the package versions used for the original publication are fixed in the file 'requirements.txt')
```
    source .venv/bin/activate
    pip install -r requirements.txt
```

3) Add the source code directory to your PYTHONPATH:
```
    export PYTHONPATH=$PROJECT_HOME/src:$PYTHONPATH
```

# Generate the results

To start the script, which sequentially generates the results shown in the paper, execute
```
    source $SUBSMIX_HOME/full_reproduction.sh
```
All output is written into subdirectories of `$SUBSMIX_HOME/output`. 

__NOTE__: Running this script in this sequential mode will take a _long_ time. It is recommended to run different parts in parallel. 
If you do so, please take into account that the order of the execution is important since some scripts expect the output of others to 
exist in `$SUBSMIX_HOME/output`. Namely, we have the following dependencies ("X→Y", means X depends on output of Y):
- `main_single_substrates.py` → `main_P_inhibens_fit.py`
- `main_P_inhibens_fit.py` → `main_P_inhibens_fit.py --omit=<run-list>`
- `main_assess_synthetic_data_fit.py` → `main_fit_synthetic_data.py <args>`




import os
import hashlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from pathlib import Path

# Control number formatting for numpy, especially important for writing TOML
np.set_printoptions(legacy="1.25")

# Pool size for main process
# NCPU = 24 # HPC
NCPU = 8 # Laptop


PROJECT_DIR = Path(os.environ.get("SUBSMIX_HOME", Path(os.environ["HOME"]) / "repos" / "subsmix"))
if not PROJECT_DIR.exists():
    raise Exception(f"Project root directory '{PROJECT_DIR}' doesn't exist. Please set environment variable 'SUBMIX_HOME' to checkout directory.")


## Initial conditions for substrate pair experiments
INITIAL_N = {1 : 5., 2 : 5.}
INITIAL_P = [.1]

# Default inoculum size
DEFAULT_V0 = 0.015

# Main directory where the experimental data resides
DATA_DIR = PROJECT_DIR / "data/experiment"

# Main directory where the experimental data resides
SYNTHETIC_DATA_DIR = PROJECT_DIR / "data/synthetic_data"

# Output directory for pickled data frames
FIT_DIR = PROJECT_DIR / "output" / "model_fits"
TMP_SAVE_DIR = PROJECT_DIR / "output" / "intermediate_saves"

# Output directory for figures
FIG_DIR = PROJECT_DIR / "output" / "figs"
TIMELINE_DIR = os.path.join(FIG_DIR, "timelines")

# Output directory for tables
TAB_DIR = PROJECT_DIR / "output" / "tables"

for d in [FIT_DIR, FIG_DIR, TAB_DIR, SYNTHETIC_DATA_DIR, TMP_SAVE_DIR]:
    if not d.exists():
        os.makedirs(d, exist_ok=True)
        print(f"Created directory '{d}'")

# Labels for the different experiment duplicates
runLabels = ["F1", "F2", "F3", "F4"]

# Directories containing the experimental data
MIX_SUBDIRS = {
    "mix11" : "mix11",
    "pure" : "individual_substrates"
    }

# Filenames for the different experimental runs
EXP_FNS = {"mix11" : dict([(i,"PInh_15mM_mix11_%s.csv"%i) for i in runLabels]),
          "pure" : dict([(i,"PInh_15mM_*_%s.csv"%i) for i in runLabels])}
      
## Figure extension for plotting
# ~ FIG_EXT = "png"
FIG_EXT = "svg"

# Colors for different components
colors = {"P" : "black",
          "N" : {1 : "brown", 2 : "orange"},
          "Npure" : {1 : "#666666", 2 : "#AAAAAA"}}

substrate_labels = ['His', 'Thr', 'Val', 'Trp', 'Phe', 'Ile', 'Leu', 'Lys', 'Glc', 'Man', 'Nag', "C ["]
pal = sns.color_palette("husl", len(substrate_labels)+1)
substrate_colors = dict([(k, c) for k,c in zip(substrate_labels, pal)])
substrate_colors["Glc"] = to_rgb("green")
substrate_colors[1] = to_rgb("brown")
substrate_colors[2] = to_rgb("orange")
substrate_colors["OD "] = to_rgb("grey")
substrate_colors["CDW"] = to_rgb("xkcd:black")
substrate_colors["E"] = to_rgb("xkcd:lavender")
substrate_colors["C"] = to_rgb("xkcd:purple")
substrate_colors["CO2"] = to_rgb("xkcd:grey blue")
substrate_colors["TDA"] = to_rgb("xkcd:burnt orange")
substrate_colors["generic"] = to_rgb("red")
substrate_colors["Man+Nag[mM]"] = to_rgb("xkcd:red")
substrate_colors["Amino acids [mM]"] = to_rgb("xkcd:blue")
substrate_colors["Glc[mM]"] = to_rgb("xkcd:green")

substrate_colors["$S_{1}$"]="#377eb8"
substrate_colors["$S_{2}$"]="#008000"
substrate_colors["$S_{3}$"]="#f781bf"
substrate_colors["$S_{4}$"]="#e41a1c"

run_colors = dict(zip(["F1","F2","F3","F4"], ["xkcd:red", "xkcd:blue", "xkcd:orange", "xkcd:purple"]))

# Ad hoc cluster colors
clusterColors = {}
# For 5 clusters (Glc isolation)
clusterColors[5] = {"Thr" : substrate_colors["Phe"],
                 "His" : substrate_colors["His"], 
                 "Val" : substrate_colors["Leu"], 
                 "Glc" : substrate_colors["Glc"],
                 "Man" : substrate_colors["Nag"]}
# For 7 clusters (Man/Nag/Thr isolation)
clusterColors[7] = {
                 "His" : substrate_colors["His"], 
                 "Trp" : substrate_colors["Trp"], 
                 "Glc" : substrate_colors["Glc"],
                 "Nag" : substrate_colors["Nag"],
                 "Man" : substrate_colors["Man"],
                 "Leu" : substrate_colors["Leu"],
                 "Val" : substrate_colors["Leu"],
                 "Thr" : substrate_colors["Thr"]}

def hash_color(obj):
    hl = hashlib.sha256(obj.encode())
    h = int(hl.hexdigest(), 16)
    col = tuple((h % p)/p for p in [983, 991, 997])
    # cols = [substrate_colors[l] for l in substrate_labels]
    # N = len(cols)
    # return cols[(h+2) % N]
    return col
    
# Line styles for different components
dashes = {"P" : "-",
          "N" : {1 : "-", 2 : "-"},
          "Npure" : {1 : "--", 2 : "-.-"}}

## Grid for plots (also determines time span of integration)
tSpan = np.linspace(0, 50, 1001)

## thetaSpan controls the initial concentration proportion \in [0.0, 1.0]
## (1 corresponds to N0 = N0[k]*NN/b[k])
# (theta=0 => pure substrate 1,
#  theta=2 => pure substrate 2)
thetaSpan = np.linspace(0.0, 1.0, 26)


## Experimental subtrate parameters (rates r, efficiencies b) for different scans
## selected by given scan-id 
rates = {}
efficiencies = {}

# scan 1: b1>b2, r1>r2
efficiencies["1a"] = {1 : 2.0, 2 : 1.0}
rates["1a"] = {1 : 1.0, 2 : 0.2}
# scan 1: b1>~b2, r1>~r2
efficiencies["1b"] = {1 : 1.1, 2 : 1.0}
rates["1b"] = {1 : 1.1, 2 : 1.0}
# scan 2a: b1>b2, r1<r2, b1*r1 > b2*r2
efficiencies["2a"] = {1 : 2.0, 2 : 1.0}
rates["2a"] = {1 : 0.75, 2 : 1.0}
# scan 2b: b1>b2, r1<r2, b1*r1 >> b2*r2
efficiencies["2b"] = {1 : 2.0, 2 : 0.25}
rates["2b"] = {1 : 0.5, 2 : 2.0}
# scan 3a: b1>b2, r1<r2, b1*r1 == b2*r2
efficiencies["3a"] = {1 : 2.0, 2 : 1.0}
rates["3a"] = {1 : 0.5, 2 : 1.0}
# scan 3b: b1>~b2, r1<~r2, b1*r1 == b2*r2
efficiencies["3b"] = {1 : 1.1, 2 : 1.0}
rates["3b"] = {1 : 1.0, 2 : 1.1}
# scan 4a: b1>b2, r1<r2, b1*r1 < b2*r2
efficiencies["4a"] = {1 : 2.0, 2 : 1.0}
rates["4a"] = {1 : 0.2, 2 : 1.0}
# scan 4b: b1>~b2, r1<<r2, b1*r1 < b2*r2
efficiencies["4b"] = {1 : 1.1, 2 : 1.0}
rates["4b"] = {1 : 0.1, 2 : 1.5}

## Half saturation constants (for Monod fct on rhs)
HFC = 1.0
hsConstants = {}
hsConstants["1a"] = {1 : HFC, 2 : HFC}
hsConstants["1b"] = {1 : HFC, 2 : HFC}
hsConstants["2a"] = {1 : HFC, 2 : HFC}
hsConstants["2b"] = {1 : HFC, 2 : HFC}
hsConstants["3a"] = {1 : HFC, 2 : HFC}
hsConstants["3b"] = {1 : HFC, 2 : HFC}
hsConstants["4a"] = {1 : HFC, 2 : HFC}
hsConstants["4b"] = {1 : HFC, 2 : HFC}
## Steepnesses of preferential associations (for preferential F)
DEFAULT_ALPHA = 1.0
prefAlphas = {}
prefAlphas["1a"] = {1 : DEFAULT_ALPHA, 2 : DEFAULT_ALPHA}
prefAlphas["1b"] = {1 : DEFAULT_ALPHA, 2 : DEFAULT_ALPHA}
prefAlphas["2a"] = {1 : DEFAULT_ALPHA, 2 : DEFAULT_ALPHA}
prefAlphas["2b"] = {1 : DEFAULT_ALPHA, 2 : DEFAULT_ALPHA}
prefAlphas["3a"] = {1 : DEFAULT_ALPHA, 2 : DEFAULT_ALPHA}
prefAlphas["3b"] = {1 : DEFAULT_ALPHA, 2 : DEFAULT_ALPHA}
prefAlphas["4a"] = {1 : DEFAULT_ALPHA, 2 : DEFAULT_ALPHA}
prefAlphas["4b"] = {1 : DEFAULT_ALPHA, 2 : DEFAULT_ALPHA}


## Abbreviations for chemicals
ABBR = {
    'Histidin':'His',
    'Glucose':'Glc',
    'Tryptophan':'Trp', 
    'Mannitol':'Man', 
    'Man':'Man',
    'Valin':'Val',
    'Leucin':'Leu', 
    'N-Acetylglucosamin':'Nag', 
    'Threonine':'Thr', 
    'Lysin':'Lys', 
    'Phenylanalin':'Phe',
    'Isoleucine':'Ile'
    }
translation = {
    'Histidin':'L-histidine',
    'Glucose':'D-glucose',
    'Tryptophan':'L-tryptophan', 
    'Mannitol':'D-mannitol',
    'Valin':'L-valine',
    'Leucin':'L-leucine', 
    'N-Acetylglucosamin':'N-acetyl-D-glucosamine', 
    'Threonine':'L-threonine', 
    'Lysin':'L-lysine', 
    'Phenylanalin':'L-phenylanaline',
    'Isoleucine':'isoleucine'
    }

subs_long_names = {a:translation[l] for l,a in ABBR.items() if l != "Man"}


## Proxy for biomass (used as P0 in fitting)
# BIOMASS_PROXY = "OD"
BIOMASS_PROXY = "CDW"


# Mix timeline figure dimensions
#timeLineDimensionsMix = (8, 2.5)
timeLineDimensionsMix = (10, 4)
# timeLineDimensions = (8, 3.5)
# timeLineDimensions = (5,2.5)
timeLineDimensions = (8, 3)
timeLineTMax = 60

# More figure dimensions
substrate_table_fig_dimensions = (7,2.25)
substrate_table_fig_dimensions2 = (10,3.5) # with title
mix_table_fig_dimensions = (4,3)

barplotInteractionsDimensions = (15, 5)


# ATP/Gipps yields for each substrate [mol ATP/(mol S_diss)]
# (using 3.3 H^+ ratio)
EnergyYields = {}
EnergyYields["ATP3.3"] = {}
EnergyYields["ATP3.3"]["Glc"] = 36.6
EnergyYields["ATP3.3"]["Man"] = 39.6
EnergyYields["ATP3.3"]["Nag"] = 47.4
EnergyYields["ATP3.3"]["Phe"] = 48.7
EnergyYields["ATP3.3"]["Trp"] = 60.6
EnergyYields["ATP3.3"]["Thr"] = 22.2
EnergyYields["ATP3.3"]["Val"] = 31.9
EnergyYields["ATP3.3"]["Leu"] = 40.7
EnergyYields["ATP3.3"]["Ile"] = 40.7
EnergyYields["ATP3.3"]["His"] = 26.6
# EnergyYields["ATP3.3"]["Lys"] = 33.4 # via cadaverine
EnergyYields["ATP3.3"]["Lys"] = 34.4 # via 2-aminoadipate
# (using 4.3 H^+ ratio)
EnergyYields["ATP4.3"] = {}
EnergyYields["ATP4.3"]["Glc"] = 28.8
EnergyYields["ATP4.3"]["Man"] = 31.1
EnergyYields["ATP4.3"]["Nag"] = 37.1
EnergyYields["ATP4.3"]["Phe"] = 38.4
EnergyYields["ATP4.3"]["Trp"] = 47.3
EnergyYields["ATP4.3"]["Thr"] = 17.4
EnergyYields["ATP4.3"]["Val"] = 24.8
EnergyYields["ATP4.3"]["Leu"] = 31.8
EnergyYields["ATP4.3"]["Ile"] = 31.8
EnergyYields["ATP4.3"]["His"] = 20.9
#EnergyYields["ATP4.3"]["Lys"] = 25.9 # via cadaverine
EnergyYields["ATP4.3"]["Lys"] = 26.9 # via 2-aminoadipate
# Gipps energy
EnergyYields["Gipps"] = {}
EnergyYields["Gipps"]["Glc"] = 2872.4
EnergyYields["Gipps"]["Man"] = 3082.5
EnergyYields["Gipps"]["Nag"] = 3786.1
EnergyYields["Gipps"]["Phe"] = 4330.8
EnergyYields["Gipps"]["Trp"] = 5016.3
EnergyYields["Gipps"]["Thr"] = 1814.0
EnergyYields["Gipps"]["Val"] = 2603.4
EnergyYields["Gipps"]["Leu"] = 3248.8
EnergyYields["Gipps"]["Ile"] = 3248.0
EnergyYields["Gipps"]["His"] = 2261.4
EnergyYields["Gipps"]["Lys"] = 3078.4

ATPKey = "ATP3.3"
ATPYields = EnergyYields[ATPKey]
substrateOrderKey = "atp" # possible keys (see utils): "growth", "alpha", "eff", "atp"
subs_names = list(ATPYields.keys())


cMolePerSubstrateMole = {}
cMolePerSubstrateMole["Glc"] = 6.0
cMolePerSubstrateMole["Man"] = 6.0
cMolePerSubstrateMole["Nag"] = 8.0
cMolePerSubstrateMole["Phe"] = 9.0
cMolePerSubstrateMole["Trp"] = 11.0
cMolePerSubstrateMole["Thr"] = 4.0
cMolePerSubstrateMole["Val"] = 5.0
cMolePerSubstrateMole["Leu"] = 6.0
cMolePerSubstrateMole["Ile"] = 6.0
cMolePerSubstrateMole["His"] = 6.0
cMolePerSubstrateMole["Lys"] = 6.0


# Whether to use a Monod functional response for the degradation of E in the GlcReservoir model
using_kE = False

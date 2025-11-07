import numpy as np
import shutil
import os
import pickle

import config

def BREAK():
    raise(Exception("BREAK"))

def orderSubstrates(substrates, params, key):
    global FIG_DIR
    
    if key.startswith("alpha"):
        substrates_ordered = [k for k in sorted(substrates)]
    elif key.startswith("atp"):
        # Sort according to ATP yield
        atp = dict([(s, config.ATPYields[s]) for s in substrates])
        substrates_ordered = [k for k in sorted(substrates, key=lambda s: atp[s])]
    elif key.startswith("eff"):
        # Sort according to yield
        y = dict([(s, params[s]["y"]) for s in substrates])
        substrates_ordered = [k for k in sorted(substrates, key=lambda s: y[s])]
    elif key.startswith("growth"):
        # Sort according to y_i*mu_i
        y = dict([(s, params[s]["y"]) for s in substrates])
        mu = dict([(s, params[s]["mu"]) for s in substrates])
        growthrate = dict([(s, y[s]*mu[s]) for s in substrates])
        substrates_ordered = [k for k in sorted(substrates, key=lambda s: growthrate[s])]
    else:
        raise("Unkown ordering '%s'"%key)
    
    return substrates_ordered


def align_yaxis_np(axes):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array(axes)
    extrema = np.array([ax.get_ylim() for ax in axes])

    # reset for divide by zero issues
    for i in range(len(extrema)):
        if np.isclose(extrema[i, 0], 0.0):
            extrema[i, 0] = -1
        if np.isclose(extrema[i, 1], 0.0):
            extrema[i, 1] = 1

    # upper and lower limits
    lowers = extrema[:, 0]
    uppers = extrema[:, 1]

    # if all pos or all neg, don't scale
    all_positive = False
    all_negative = False
    if lowers.min() > 0.0:
        all_positive = True

    if uppers.max() < 0.0:
        all_negative = True

    if all_negative or all_positive:
        # don't scale
        return

    # pick "most centered" axis
    res = abs(uppers+lowers)
    min_index = np.argmin(res)

    # scale positive or negative part
    multiplier1 = abs(uppers[min_index]/lowers[min_index])
    multiplier2 = abs(lowers[min_index]/uppers[min_index])

    for i in range(len(extrema)):
        # scale positive or negative part based on which induces valid
        if i != min_index:
            lower_change = extrema[i, 1] * -1*multiplier2
            upper_change = extrema[i, 0] * -1*multiplier1
            if upper_change < extrema[i, 1]:
                extrema[i, 0] = lower_change
            else:
                extrema[i, 1] = upper_change

        # bump by 10% for a margin
        extrema[i, 0] *= 1.1
        extrema[i, 1] *= 1.1

    # set axes limits
    [axes[i].set_ylim(*extrema[i]) for i in range(len(extrema))]
    
def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])
    

def saveState(params, fn):
    # secure saving (makes backup before writing to file)
    if os.path.exists(fn):
        shutil.copy(fn, str(fn)+".bak")
    with open(fn, "wb") as f:
        pickle.dump(params, f)
    
def loadState(fn):
    with open(fn, "rb") as f:
        params = pickle.load(f)
    return params


def mergeDicts(d1, d2, maxdepth=100):
    ''' Copies all values from d1 into d2
    '''
    if maxdepth==0:
        raise("maximum depth in mergeDicts()")
    for k, v in d1.items():
        if k in d2 and isinstance(d2[k], dict):
            d2[k] = mergeDicts(d1[k], d2[k], maxdepth=maxdepth-1)
        else:
            # Write v into d2, might override present d2[k]
            d2[k] = v
    return d2
    
    
def mergeStates(fn_in1, fn_in2, fn_out):
    ''' Copies all content of results dict in fn_in1 
        into results dict from fn_in2 and writes it into fn_out
    '''
    params1 = loadState(fn_in1)
    params2 = loadState(fn_in2)
    params = mergeDicts(params1, params2)    
    with open(fn_out, "wb") as f:
        pickle.dump(params, f)
    return params
    




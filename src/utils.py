import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import seaborn as sns
from copy import deepcopy 
from time import time
from tqdm import tqdm
from itertools import tee
import pandas as pd
import sys

def decompose(cpt):

    from itertools import tee
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


    p_start  = [a for a, _ in pairwise(cpt)]
    p_end  = [b for _, b in pairwise(cpt)]
    return p_start, p_end
    
    
def join_and_discard(frame, join_len, discard_len, binary_mask):
    """
    This function applies smoothing to frame such that:
    1) Gaps smaller or equal to "join_len" are filled,
    2) Isolated continuous frames with length smaller than "discard_len" are removed
    
    Arguments:
        frame {[int]} -- The indices
        join {int} -- threshold for joining discrete groups
        discard {int} -- threshold for discarding groups
    """
    
    if binary_mask:
        original_size=len(frame)
        frame = np.argwhere(frame==1).squeeze()
        
        
    # print(frame)
    if len(frame) == 0:
        return frame
    
    # First join
    frame = sorted(frame)
    joined_frame = []
    prev = frame[0]
    for f in frame[1:]:
        if f > prev + 1 and f < prev + join_len:
            joined_frame.extend(list(range(prev+1, f)))
        prev = f

    frame = sorted(frame + joined_frame)
    # print(joined_frame)
    
    # Then discard
    discard_frame = []
    prev = frame[0]
    island = [prev]
    for f in frame[1:]:
        if f == prev + 1:
            island.append(f)
        else:
            # check island length
            if len(island) < discard_len:
                discard_frame.extend(island)
            island = [f]
        prev = f

    if len(island) < discard_len:
        discard_frame.extend(island)

    # print(discard_frame)
    new_frame = [f for f in frame if f not in discard_frame]
    
    
    if binary_mask:
        new_mask = np.zeros(original_size)
        new_mask[new_frame] = 1
        return new_mask
    
    return new_frame

def fi(x=25, y=4):
    return plt.figure(figsize=(x,y))

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


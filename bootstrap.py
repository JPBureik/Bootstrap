#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:44:36 2024

@author: jp
"""

# Standard library imports:
import numpy as np
from tqdm import tqdm
import boost_histogram as bh
from matplotlib import pyplot as plt

# Local imports:
from mcpmeas.helper_functions import multiproc_list


def bootstrap_err(func, sample, conf_int=15.9, n_bs=100000, parallel=False,
                  desc='Bootstrapping Errorbars', plot_bs_distr=False):
    bs_dist = np.zeros(n_bs)
    
    sample_func_tuple = (sample, func)
    
    def single_pseudo_sample(sample_func_tuple):
        sample, func = sample_func_tuple
        pseudo_sample = np.random.choice(sample, size=sample.size,
                                         replace=True)
        return func(pseudo_sample)
    
    arg_array = np.empty(n_bs, dtype=object)
    arg_array[:] = [sample_func_tuple] * n_bs
    
    if parallel:
        bs_dist = multiproc_list(
            arg_array,
            single_pseudo_sample,
            parallelize='workers',
            show_pbar=True,
            desc=desc
            )
    else:
        bs_dist = np.array([single_pseudo_sample(arg)
                            for arg in tqdm(arg_array, desc=desc)])
    
    if plot_bs_distr:
        
        hist = bh.Histogram(bh.axis.Regular(int(np.floor(min(bs_dist))), int(np.ceil(max(bs_dist))), int(np.rint(np.sqrt(n_bs)))))
        hist.fill(bs_dist)
        
        fig, ax = plt.subplots()
        ax.bar(
            hist.axes[0].centers, hist.values(), width=hist.axes[0].widths
            )
        plt.show()
    
    
    # Set level of confidence to 68.2%:
    return np.mean([abs(func(sample) - bound) for bound in np.percentile(bs_dist, [conf_int, 100 - conf_int])])
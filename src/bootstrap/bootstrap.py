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


def bootstrap_ci(sample, func, ci_range=15.9, n_bs=10000, parallel=False,
                 plot_bs_distr=False):
    r"""
    Bootstrap confidence intervals.
    
    Create bootstrap samples from sample on which to evaluate func and return
    the range specified by ci_range in the resulting histogram as confidence
    intervals.

    Parameters
    ----------
    sample: ndarray of int or float or pandas.DataFrame, dtype: int
        Data  
    func: function object
        The confidence intervals are evaluated for the result of func(sample).
    ci_range: int or float
        Range of the onfidence interval. Default: 15.9 (corresponds to 1 sigma
        for normal distributions).
    n_bs: int
        Number of boostrap samples. Default: 10000
    parallel: bool
        Parallelize evaluation of boostrap samples over all physical CPU
        cores. Default: False.
    plt_bs_distr: bool
        Plot the histogram of func(sample) for all bootstrap samples. Default:
        False.
    
    Returns
    -------
    ci: ndarray of int or float
        Confidence interval.
    """
    
    # Initialize bootstrap distribution:
    bs_dist = np.zeros(n_bs)
    
    # Bundle sample and function to single argument for multiprocessing:
    sample_func_tuple = (sample, func)
    
    def single_pseudo_sample(sample_func_tuple):
        r"""Helper function for multiprocessing.
        
        Create a new bootstrap sample from the original sample and return the
        result of calling func on it.
        
        Parameters
        ----------
        sample_func_tuple: tuple of ndarray and func
            Single object bundling the original sample and the function.
        
        Returns
        -------
        func(pseudo_sample): int or float
            The result of calling func on the generated bootstrap sample.
        """
        
        # Unpack tuple:
        sample, func = sample_func_tuple
        
        # Generate bootstrap sample:
        pseudo_sample = np.random.choice(sample, size=sample.size,
                                         replace=True)
        # Call func on bootstrap sample:
        return func(pseudo_sample)
    
    # Initialize data container:
    arg_array = np.empty(n_bs, dtype=object)
    arg_array[:] = [sample_func_tuple] * n_bs
    
    if parallel:
        bs_dist = multiproc_list(
            arg_array,
            single_pseudo_sample,
            parallelize='workers',
            show_pbar=True,
            desc='Bootstrapping Errorbars'
            )
    else:
        bs_dist = np.array([single_pseudo_sample(arg)
                            for arg in tqdm(arg_array,
                                            desc='Bootstrapping Errorbars')])
    ci = {
        'lower': np.percentile(bs_dist, ci_range),
        'upper': np.percentile(bs_dist, 100 - ci_range)
            }
    
    if plot_bs_distr:
        
        hist = bh.Histogram(bh.axis.Regular(int(np.rint(np.sqrt(n_bs))), min(bs_dist), max(bs_dist)))
        hist.fill(bs_dist)
        
        fig, ax = plt.subplots()
        ax.bar(
            hist.axes[0].centers, hist.values(), width=hist.axes[0].widths,
            color='steelblue',
            label='Function: ' + func.__name__
            )
        for limit, r in zip(ci, [ci_range, 100 - ci_range]):
            ax.axvline(
                ci[limit],
                color='indianred',
                linestyle='--',
                label=str(r) + '%: ' + str(np.round(ci[limit], 1))
                )
        ax.axvline(
            func(sample),
            color='k',
            label='Value: ' + str(np.round(func(sample), 1))
            )
        ax.legend()
        plt.show()
    

    # Set level of confidence to 68.2%
    return ci

#%%

if __name__ == '__main__':
    
    mu, sigma = 1, 100 # mean and standard deviation
    s = np.random.normal(mu, sigma, 10000)
    ci = bootstrap_ci(s, np.var, parallel=True, plot_bs_distr=True)


#!/usr/bin/env python -W ignore::DeprecationWarning

import os
import numpy as np

from ss_plotting.make_plots import plot


if __name__ == '__main__':
    files = (['env_shelf01', 'env_table1', 'env_table3',
              'env_shelf02', 'env_kitchen1', 'env_kitchen2',
              'env_kitchen_refrigerator', 'env_kitchen_microwave'])  

    local_path = os.getcwd()
    results_path = os.path.join(local_path, "metric_results/")

    k_values = [1,5,10,15,20]
    N_values = [1000,5000,10000,15000,20000]
    r_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
    n = np.load(os.path.join(results_path, 'avg_krnl.npz'))
    accnn = n['accg']
    accann = n['accep']
    errnn = n['errg']
    errann = n['errep']    
        
    series_colors = ['red', 'blue']
    series_labels = ['accuracy', 'error']
    fontsize = 16
    linewidth = 2
    ylbl = 'Metric'
    legend_loc = 'center right'
    ylim = (0.0,0.9)
    xsize = 3.4
    ysize = 1.5
    amp = 2.5
    picsize = (amp*xsize, amp*ysize)
    
    # change these accordingly
    title = 'Epanechnikov Kernel'
    xlbl = 'No. of samples'
    xval = N_values
    xlim = (1.0, xval[-1])
    
    # change these accordingly
    acc = accann[:,1]
    err = errann[:,1]
    series_line = [(xval, acc), (xval, err)]
    
    plot(
        series_line, 
        series_colors=series_colors, 
        series_labels=series_labels, 
        plot_xlabel=xlbl,
        plot_ylabel=ylbl, 
        plot_title=title, 
        linewidth=linewidth, 
        fontsize=fontsize, 
        legend_fontsize=fontsize, 
        legend_location=legend_loc, 
        plot_ylim=ylim, 
        plot_xlim=xlim,
        # show_plot = True,
        savefile='EP_r=15' + '.jpg',
        savefile_size=picsize
    )


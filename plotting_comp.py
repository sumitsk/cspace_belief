#!/usr/bin/env python -W ignore::DeprecationWarning

import os
import numpy

from ss_plotting.make_plots import plot


if __name__ == '__main__':

    local_path = os.getcwd()
    results_path = os.path.join(local_path, "metric_results/")

    # k_values = [1,5,10,15,20]
    N_values = [1000,5000,10000,15000,20000]
    # r_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
    n = numpy.load(os.path.join(results_path, 'avg_NN.npz'))
    accnn = n['accnn']
    accann = n['accann']
    errnn = n['errnn']
    errann = n['errann']    
        
    nw = numpy.load(os.path.join(results_path, 'avg_NN_w.npz'))
    accnnw = nw['accnn']
    accannw = nw['accann']
    errnnw = nw['errnn']
    errannw = nw['errann']    
    
    series_colors = ['red', 'blue', 'green', 'purple']
    series_labels = ['Euclidean accuracy', 'Weighted Euclidean accuracy', 'Euclidean error', 'Weighted Euclidean error']
    fontsize = 8
    linewidth = 2
    ylbl = 'Metric'
    # legend_loc = 'upper right'
    ylim = (0.2,0.9)
    xsize = 3.4
    ysize = 1.5
    amp = 2.5
    picsize = (amp*xsize, amp*ysize)
    
    # change these accordingly
    title = 'Approximate Nearest Neighbors'
    xlbl = 'No. of samples'
    xval = N_values
    xlim = (0, xval[-1])
    
    # change these accordingly
    acc = accann[:,2]
    accw = accannw[:,2]
    err = errann[:,2]
    errw = errannw[:,2]
    
    series_line=[(xval, acc), (xval, accw), (xval, err), (xval, errw) ]   
    plot(series_line, series_colors=series_colors, 
    series_labels=series_labels, 
    plot_xlabel=xlbl,
    plot_ylabel=ylbl, 
    plot_title=title, 
    linewidth=linewidth,  
    fontsize=fontsize, 
    legend_fontsize=fontsize, 
    # legend_location=legend_loc, 
    plot_ylim=ylim, 
    plot_xlim=xlim,
    show_plot=True,
    savefile='ANN_weights_comp_k=10' + '.jpg',
    savefile_size=picsize)
    

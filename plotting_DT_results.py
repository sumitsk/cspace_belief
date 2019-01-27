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
    
    nd = numpy.load(os.path.join(results_path, 'avg_DT.npz'))
    accdt = nd['acc']
    errdt = nd['err']
        
    nd = numpy.load(os.path.join(results_path, 'avg_DT.npz'))
    accdt = nd['acc']
    accdtw = nd['accw']
    accdtm = nd['accm']
    accdtmw = nd['accmw']
    errdt = nd['err']
    errdtw = nd['errw']
    errdtm = nd['errm']
    errdtmw = nd['errmw']
    
    nk = numpy.load(os.path.join(results_path, 'avg_krnl.npz'))
    accg = nk['accg']
    accep = nk['accep']
    errg = nk['errg']
    errep = nk['errep']    
    
    series_colors = ['red', 'blue', 'green', 'purple', 'red', 'blue', 'green', 'purple']
    series_labels = ['E accuracy', 'WE accuracy', 'M accuracy', 'WM accuracy', 'E error', 'WE error', 'M error', 'WM error']
    line_styles = ['-', '-', '-', '-', '--', '--', '--', '--']
    fontsize = 16
    linewidth = 2
    ylbl = 'Metric'
    # legend_loc = 'upper right'
    ylim = (0.0,1.0)
    xsize = 3.4
    ysize = 1.5
    amp = 2.5
    picsize = (amp*xsize, amp*ysize)
    
    # change these accordingly
    title = 'Topological Method'
    xlbl = 'No. of samples'
    xval = N_values
    xlim = (0, xval[-1])
    
    # change these accordingly
    
    accann = accann[:,2]
    accnn = accnn[:,2]
    accg = accg[:,1]
    accep = accep[:,1]
    
    errnn = errnn[:,2]
    errann = errann[:,2]
    errep = errep[:,1]
    errg = errg[:,1]
    # errw = errannw[:,2]
    
    series_line = [(xval, accdt), (xval, accdtw), (xval, accdtm), (xval, accdtmw),
                   (xval, errdt), (xval, errdtw), (xval, errdtm), (xval, errdtmw)]   
    
    plot(series_line, series_colors=series_colors, 
    series_labels=series_labels, 
    plot_xlabel=xlbl,
    plot_ylabel=ylbl, 
    plot_title=title, 
    linewidth=linewidth, 
    fontsize=fontsize, 
    legend_fontsize=fontsize,
    line_styles=line_styles, 
    # legend_location=legend_loc, 
    plot_ylim=ylim, 
    plot_xlim=xlim,
    show_plot=True,
    savefile='DT_all' + '.jpg',
    savefile_size=picsize)
    

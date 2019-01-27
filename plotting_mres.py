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
    
    nm = numpy.load(os.path.join(results_path, 'avg_NN_m.npz'))
    accnnm = nm['accnn']
    accannm = nm['accann']
    errnnm = nm['errnn']
    errannm = nm['errann']    
        
    nmw = numpy.load(os.path.join(results_path, 'avg_NN_m_w.npz'))
    accnnmw = nmw['accnn']
    accannmw = nmw['accann']
    errnnmw = nmw['errnn']
    errannmw = nmw['errann'] 
    
    series_colors = ['red', 'blue', 'green', 'purple', 'red', 'blue', 'green', 'purple']
    series_labels = ['E accuracy', 'WE accuracy', 'M accuracy', 'WM accuracy', 'E error', 'WE error', 'M error', 'WM error']
    line_styles = ['-', '-', '-', '-', '--', '--', '--', '--']
    fontsize = 16
    linewidth = 2
    ylbl = 'Metric'
    # legend_loc = 'upper right'
    ylim = (0.2,0.9)
    xsize = 3.4
    ysize = 1.5
    amp = 2.5
    picsize = (amp*xsize, amp*ysize)
    
    # change these accordingly
    title = 'Nearest Neighbours'
    xlbl = 'No. of samples'
    xval = N_values
    xlim = (0, xval[-1])
    
    # change these accordingly
    acc = accnn[:,2]
    accw = accnnw[:,2]
    accm = accnnm[:,2]
    accmw = accnnmw[:,2]
    err = errnn[:,2]
    errw = errnnw[:,2]
    errm = errnnm[:,2]
    errmw = errnnmw[:,2]
    
    series_line = [(xval, acc), (xval, accw), (xval, accm), (xval, accmw), (xval, err),
                   (xval, errw), (xval, errm), (xval, errmw)]   
    plot(series_line, series_colors=series_colors, 
    series_labels=series_labels, 
    line_styles=line_styles,
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
    savefile='NN_all_k=10' + '.jpg',
    savefile_size=picsize)
    

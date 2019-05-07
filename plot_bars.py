import os
import numpy as np
from ss_plotting.make_plots import plot_bar_graph


if __name__ == '__main__':

    files = (['env_shelf01', 'env_table1', 'env_table3',
              'env_shelf02', 'env_kitchen1', 'env_kitchen2',
              'env_kitchen_refrigerator', 'env_kitchen_microwave'])  

    local_path = os.getcwd()
    path = os.path.join(local_path, 'metric_results')
    save_path = os.path.join(local_path, 'ss_plots')
    
    k_values = [1,5,10,15,20]
    N_values = [1000,5000,10000,15000,20000]
    r_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
    series_colors = ['red', 'blue']
    series_labels = ['accuracy', 'error']
    fontsize = 8
    linewidth = 2
    ylbl = 'Metric'
    legend_loc = 'upper right'
    ylim = (0,1)
    xsize = 3.4
    ysize = 1.5
    amp = 2
    picsize = (amp*xsize, amp*ysize)
    title = "Topological Method"
    names = ['.npz', '_m.npz', '_w.npz', '_m_w.npz']
    
    acc = []
    err = []
    
    for l in names:
        fn = 'avg_NN' + l
        n = np.load(os.path.join(path,fn))
        acc.append(n['accnn'][2][2])
        err.append(n['errnn'][2][2])
    
    directory = os.path.join(save_path)
        
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)
        
    categories = ['e', 'm', ' we', 'wm']
    series = [acc, err]
    plot_bar_graph(
        series, 
        series_colors,
        series_labels=series_labels,
        category_labels=categories,
        barwidth=0.25,
        plot_ylabel=ylbl, 
        plot_title=title,
        legend_fontsize=fontsize/2,
        fontsize=fontsize,
        savefile='DT.jpg',
        savefile_size=picsize
    )


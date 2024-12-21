# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import muspan as ms
import os

# Set seaborn theme for better visualization
sns.set_theme(style='white', font_scale=1)

# Define the save path for images
savepath = './Images/'

# Get the directory path of the current script
dir_path = os.path.dirname(os.path.abspath(__file__))
print(dir_path)

# Load the domain from a .muspan file
domain = ms.io.load_domain(dir_path + '/Weeratunga_2023_example.muspan')

# Visualize the domain with cell types
ms.visualise.visualise(domain, 'Celltype', marker_size=10)

# Define pairs of cell types to analyze
to_plot = ['Blood Vessels - UD structural SM', 'UD structural SM - Myofibroblast']

# Set random seed for reproducibility
np.random.seed(0)

# Loop through each pair of cell types
for pair in to_plot:
    if pair == 'Blood Vessels - UD structural SM':
        q1 = ms.query.query(domain, ('label', 'Celltype'), 'is', 'Blood vessels')
        q2 = ms.query.query(domain, ('label', 'Celltype'), 'is', 'UD structural SM')
    elif pair == 'UD structural SM - Myofibroblast':
        q1 = ms.query.query(domain, ('label', 'Celltype'), 'is', 'UD structural SM')
        q2 = ms.query.query(domain, ('label', 'Celltype'), 'is', 'Myofibroblast')
    
    q3 = q1 | q2

    # Visualize all cells
    ms.visualise.visualise(domain, 'Celltype', marker_size=20)
    #plt.savefig(f'{savepath}{pair} - 1 - all cells.png')
    #plt.savefig(f'{savepath}{pair} - 1 - all cells.svg')

    # Visualize only the selected cells
    ms.visualise.visualise(domain, 'Celltype', marker_size=20, objects_to_plot=q3)
    #plt.savefig(f'{savepath}{pair} - 1 - some cells.png')
    #plt.savefig(f'{savepath}{pair} - 1 - some cells.svg')

    # Nearest neighbours analysis
    d, oia, nearest_B, verts_a, verts_b = ms.query.get_minimum_distances_centroids(domain, q1, q2)
    ms.visualise.visualise(domain, 'Celltype', objects_to_plot=q3, marker_size=20, show_boundary=True)
    
    for i in range(len(verts_a)):
        p1 = verts_a[i]
        p2 = verts_b[i]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c=[1, 0, 0], linestyle='-', lw=5)
    
    plt.savefig(f'{savepath}{pair} - 2 - nearest neighbours.png')
    plt.savefig(f'{savepath}{pair} - 2 - nearest neighbours.svg')

    # Quadrant Correlation Matrix (QCM)
    rkw = {'side_length': 100}
    SES, A, label_categories = ms.region_based.quadrat_correlation_matrix(domain, 'Celltype', verbose=True, region_kwargs=rkw)
    ms.visualise.visualise_correlation_matrix(A, label_categories, triangle_to_plot='lower', colorbar_limit=[-5, 5])
    plt.tight_layout()
    #plt.savefig(f'{savepath}{pair} - 3 - QCM.png')
    #plt.savefig(f'{savepath}{pair} - 3 - QCM.svg')

    # Pair Correlation Function (PCF) and K-function
    sns.set_theme(style='white', font_scale=3)
    rk, K = ms.spatial_statistics.cross_k_function(domain, q1, q2, max_R=300, step=10)
    r, PCF = ms.spatial_statistics.cross_pair_correlation_function(domain, q1, q2, max_R=300, annulus_step=10, annulus_width=20)
    
    plt.figure(figsize=(12, 8))
    plt.plot(r, PCF, lw=10)
    plt.gca().axhline(1, c='k', linestyle=':', lw=7)
    plt.ylim([0, 5])
    plt.ylabel('$g(r)$')
    plt.xlabel('$r$')
    #plt.savefig(f'{savepath}{pair} - 4 - PCF.png')
    #plt.savefig(f'{savepath}{pair} - 4 - PCF.svg')
    
    plt.figure(figsize=(12, 8))
    plt.plot(rk, K, lw=10)
    plt.plot(rk, np.pi * rk**2, c='k', linestyle=':', lw=7)
    plt.ylabel('$K(r)$')
    plt.xlabel('$r$')
    #plt.savefig(f'{savepath}{pair} - 5 - K.png')
    #plt.savefig(f'{savepath}{pair} - 5 - K.svg')

    # Topographical Correlation Map (TCM)
    TCM = ms.spatial_statistics.topographical_correlation_map(domain, q1, q2)
    ms.visualise.visualise_topographical_correlation_map(domain, TCM)
    ms.visualise.visualise(domain, 'Celltype', objects_to_plot=q3, marker_size=20, show_boundary=False, ax=plt.gca(), add_cbar=False)
    #plt.savefig(f'{savepath}{pair} - 6 - TCM.png')
    #plt.savefig(f'{savepath}{pair} - 6 - TCM.svg')

    # Network analysis
    sns.set_theme(style='white', font_scale=1)
    del_network = ms.networks.generate_network(domain, objects_as_nodes=q3, network_name='Delaunay CC', network_type='Delaunay', max_edge_distance=50)
    ms.visualise.visualise_network(domain, network_name='Delaunay CC', visualise_kwargs={'color_by': 'Celltype', 'objects_to_plot': q3, 'marker_size': 5}, add_cbar=False, edge_width=3)
    plt.savefig(f'{savepath}{pair} - 7 - network.png')
    plt.savefig(f'{savepath}{pair} - 7 - network.svg')
    
    ms.visualise.visualise_network(domain, network_name='Delaunay CC', visualise_kwargs={'color_by': 'Celltype', 'objects_to_plot': q3, 'marker_size': 50, 'add_scalebar': True}, add_cbar=False, edge_width=5)
    plt.xlim([500, 1000])
    plt.ylim([650, 1150])
    #plt.savefig(f'{savepath}{pair} - 8 - network zoom.png')
    #plt.savefig(f'{savepath}{pair} - 8 - network zoom.svg')

    # Wasserstein distance analysis
    kde1 = ms.distribution.kernel_density_estimation(domain, q1, visualise_output=False)
    kde2 = ms.distribution.kernel_density_estimation(domain, q2, visualise_output=False)
    
    if pair == 'Blood Vessels - UD structural SM':
        c1 = 'Blues'
        c2 = 'Greens'
    else:
        c1 = 'Greens'
        c2 = 'bone_r'
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ms.visualise.visualise_heatmap(domain, kde1, ax=axes[0], heatmap_cmap=c1)
    ms.visualise.visualise_heatmap(domain, kde2, ax=axes[1], heatmap_cmap=c2)
    plt.tight_layout()
    #plt.savefig(f'{savepath}{pair} - 9 - KDE.png')
    #plt.savefig(f'{savepath}{pair} - 9 - KDE.svg')

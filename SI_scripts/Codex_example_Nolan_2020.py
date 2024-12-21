# Import necessary libraries
import muspan as ms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import os

# Set seaborn theme for plots
sns.set_theme(style='ticks', font_scale=2)

# Set font properties for scalebar
fontprops = fm.FontProperties(size=18)

# Define the path to save images
savepath = './Images/'

# Get the file path

# DOWNLOAD THE DATA FROM THIS LINK: https://data.mendeley.com/datasets/mpjzbtfgfr/1
# IT'S THE FILE CALLED 'CRC_clusters_neighborhoods_markers.csv'
dir_path = 'path/to/where/you/save/the/data'

# Read in the data made available by Schuerch et al 2020
data = pd.read_csv(dir_path+'/CRC_clusters_neighborhoods_markers.csv')

# Define the cell order and colors to match Schuerch et al
cell_order = ['tumor cells', 'CD11c+ DCs', 'tumor cells / immune cells',
              'smooth muscle', 'lymphatics', 'adipocytes', 'undefined',
              'CD4+ T cells CD45RO+', 'CD8+ T cells', 'CD68+CD163+ macrophages',
              'plasma cells', 'Tregs', 'immune cells / vasculature', 'stroma',
              'CD68+ macrophages GzmB+', 'vasculature', 'nerves',
              'CD11b+CD68+ macrophages', 'granulocytes', 'CD68+ macrophages',
              'NK cells', 'CD11b+ monocytes', 'immune cells',
              'CD4+ T cells GATA3+', 'CD163+ macrophages', 'CD3+ T cells',
              'CD4+ T cells', 'B cells', 'dirt']

cell_colors = [sns.color_palette('tab20')[i % 20] for i in range(len(cell_order))]
cmap_dict = {cell_order[v]: cell_colors[v] for v in range(len(cell_order))}

# Plot the data as muspan domains
for group in [1, 2]:
    ncols = 10
    nrows = 7 if group == 1 else 8
    fig, ax = plt.subplots(figsize=(20, 17), nrows=nrows, ncols=ncols, gridspec_kw={'hspace': 0.1, 'wspace': 0.1})

    spot_labs = np.unique(data['spots'])
    i_num = -1
    for pat_index in spot_labs:
        mask = (data['spots'] == pat_index) & (data['groups'] == group)
        points = np.asarray([data['X:X'][mask], data['Y:Y'][mask]]).T
        labels = list(data['ClusterName'][mask])
        
        # Remove duplicate points
        unique_points, unique_labels = [], []
        seen = set()
        for point, label in zip(points, labels):
            point_tuple = tuple(point)
            if point_tuple not in seen:
                unique_points.append(point)
                unique_labels.append(label)
                seen.add(point_tuple)
        points = np.array(unique_points)
        labels = unique_labels
        
        if len(points) > 0:
            i_num += 1
            pc = ms.domain('TMA')
            pc.add_points(points, 'Cell centres')
            pc.add_labels('Celltype', labels, 'Cell centres')
            new_colors = {j: cell_colors[cell_order.index(j)] for j in pc.labels['Celltype']['categories']}
            pc.update_colors(new_colors, colors_to_update='labels', label_name='Celltype')
            ms.visualise.visualise(pc, color_by=('label', 'Celltype'), ax=ax[int(np.floor(i_num / ncols)), i_num % ncols], marker_size=2, add_cbar=False)
            scalebar = AnchoredSizeBar(ax[int(np.floor(i_num / ncols)), i_num % ncols].transData, 250, '', 'lower right', pad=2, color='black', frameon=False, size_vertical=20, fontproperties=fm.FontProperties(size=5), zorder=100)
            ax[int(np.floor(i_num / ncols)), i_num % ncols].add_artist(scalebar)
            ax[int(np.floor(i_num / ncols)), i_num % ncols].set_aspect('equal')
            ax[int(np.floor(i_num / ncols)), i_num % ncols].set_xticks([])
            ax[int(np.floor(i_num / ncols)), i_num % ncols].set_yticks([])

    # Hide the last few unused axes to make sure the plots aren't cluttered
    n_axes_to_hide = {1: 2, 2: 8}
    for i in range(1, n_axes_to_hide[group] + 1):
        ax[nrows - 1, ncols - i].axis('off')

# Standalone colorbar
from matplotlib.lines import Line2D

def build_legend(data):
    """
    Build a legend for matplotlib plt from dict
    """
    legend_elements = []
    for key in data:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=key,
                                      markerfacecolor=data[key], markersize=10))
    return legend_elements

fig, ax = plt.subplots(1)
legend_elements = build_legend(cmap_dict)
ax.legend(handles=legend_elements, loc='upper left')
plt.show()

# Reproduce the Schurch/Nolan pipeline
data = pd.read_csv('Nolan_2020_data.csv')
uniqueLabs = np.unique(data['ClusterName'])

spot_labs = np.unique(data['spots'])
list_of_domains = []

for pat_index in spot_labs:
    for group in np.unique(data['groups']):
        mask = (data['spots'] == pat_index) & (data['groups'] == group)
        points = np.asarray([data['X:X'][mask], data['Y:Y'][mask]]).T
        labels = list(data['ClusterName'][mask])
        proximity_cluster_labels = list(data['neighborhood10'][mask])

        if len(points) > 0:
            pc = ms.domain('TMA - ' + str(pat_index) + ' - ' + str(group))
            pc.add_points(points, 'Cell centres')
            pc.add_labels('Celltype', labels, 'Cell centres')
            pc.add_labels(label_name='Schürch 2020', labels=proximity_cluster_labels, add_labels_to=None, cmap='Set3')
            list_of_domains.append(pc)

# Do neighbourhood enrichment, and plot the outcome
neighbourhood_enrichment_matrix, consistent_global_labels, unique_cluster_labels = ms.networks.cluster_neighbourhoods(
    list_of_domains, label_name='Celltype', populations_to_analyse=None, network_type='KNN',
    force_labels_to_include=uniqueLabs, labels_to_ignore=['dirt'], k_hops=1, max_edge_distance=np.inf, min_edge_distance=0,
    number_of_nearest_neighbours=9, neighbourhood_label_name='Neighbourhood ID', cluster_method='minibatchkmeans', cluster_parameters={'n_clusters': 10, 'random_state': 0})

df_ME_id = pd.DataFrame(data=neighbourhood_enrichment_matrix, index=unique_cluster_labels, columns=consistent_global_labels)
df_ME_id.index.name = 'ME ID'
df_ME_id.columns.name = 'Celltype ID'

# Plot the neighbourhood enrichment matrix
sns.clustermap(df_ME_id, xticklabels=consistent_global_labels, yticklabels=unique_cluster_labels, figsize=(10, 3.5), cmap='seismic', dendrogram_ratio=(.05, .3), col_cluster=True, row_cluster=True, square='True',
               linewidths=0.5, linecolor='black', cbar_kws=dict(use_gridspec=False, location="right", label='ME enrichment', ticks=[-3, 0, 3]), cbar_pos=(0.12, 1.75, 0.72, 0.08),
               vmin=-3, vmax=3, tree_kws={'linewidths': 0, 'color': 'white'})

# Plot specific domains with neighbourhood and Schürch 2020 labels
to_plot = [10, 20, 30, 40, 50]
for show_pc_ind in to_plot:
    # Faff with colours
    cats = list_of_domains[show_pc_ind].labels['Neighbourhood ID']['categories']
    col_dict_neighbourhood = {v: plt.cm.tab10(v) for v in cats}
    cats = list_of_domains[show_pc_ind].labels['Schürch 2020']['categories']
    col_dict_nolan = {v: plt.cm.Set3(v) for v in cats}

    list_of_domains[show_pc_ind].update_colors(col_dict_neighbourhood, label_name='Neighbourhood ID')
    list_of_domains[show_pc_ind].update_colors(col_dict_nolan, label_name='Schürch 2020')

    fig, ax = plt.subplots(figsize=(20, 7), nrows=1, ncols=2)
    ms.visualise.visualise(list_of_domains[show_pc_ind], color_by=('label', 'Neighbourhood ID'), ax=ax[0])
    ax[0].set_title('MuSpAn')
    ms.visualise.visualise(list_of_domains[show_pc_ind], color_by=('label', 'Schürch 2020'), ax=ax[1])
    ax[1].set_title('Schürch 2020')
    # plt.savefig(f'{savepath}Comparison_{show_pc_ind}.png')
    # plt.savefig(f'{savepath}Comparison_{show_pc_ind}.svg')

    # Convert point-like data into Voronoi polygons
    domain = list_of_domains[show_pc_ind]
    domain.estimate_boundary(method='alpha shape', alpha_shape_kwargs={'alpha': 1000})
    # ms.visualise.visualise(domain, show_boundary=True)
    domain.convert_objects(
        population=None,
        object_type='shape',
        conversion_method='voronoi',
        collection_name='Estimated boundaries',
        inherit_collections=False
    )
    domain.update_colors(col_dict_neighbourhood, label_name='Neighbourhood ID')
    domain.update_colors(col_dict_nolan, label_name='Schürch 2020')
    fig, ax = plt.subplots(figsize=(20, 7), nrows=1, ncols=2)
    ms.visualise.visualise(list_of_domains[show_pc_ind], color_by=('label', 'Neighbourhood ID'), ax=ax[0], objects_to_plot=('collection', 'Estimated boundaries'))
    ax[0].set_title('MuSpAn')
    ms.visualise.visualise(list_of_domains[show_pc_ind], color_by=('label', 'Schürch 2020'), ax=ax[1], objects_to_plot=('collection', 'Estimated boundaries'))
    ax[1].set_title('Schürch 2020')
    plt.show()

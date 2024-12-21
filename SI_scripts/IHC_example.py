# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import muspan as ms

# Set the theme for seaborn
sns.set_theme(style='white', font_scale=2)

# Load the example domain data
domain = ms.datasets.load_example_domain('Macrophage-Hypoxia-ROI')

# Visualize the loaded domain
ms.visualise.visualise(domain)

# Calculate the KDE for all macrophages across the entire domain
kde_all = ms.distribution.kernel_density_estimation(
    domain,
    population=('Collection', 'Macrophages'),
    mesh_step=5
)

# Calculate the KDE for macrophages within the tumour boundaries
kde_tumour = ms.distribution.kernel_density_estimation(
    domain,
    population=('Collection', 'Macrophages'),
    include_boundaries=('Collection', 'PanCK'),
    mesh_step=5
)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Visualize the KDE for all macrophages
ms.visualise.visualise_heatmap(domain, kde_all, ax=axes[0], heatmap_cmap='Blues', colorbar_limit=[0, 1e-6])
axes[0].set_title('All macrophages')

# Visualize the KDE for macrophages within the tumour
ms.visualise.visualise_heatmap(domain, kde_tumour, ax=axes[1], heatmap_cmap='Blues', colorbar_limit=[0, 1e-6])
axes[1].set_title('Macrophages in tumour only')

# Overlay the PanCK boundaries
for ax in axes:
    ms.visualise.visualise(domain, objects_to_plot=('collection', 'PanCK'), ax=ax, shape_kwargs={'edgecolor': 'k', 'fill': False, 'alpha': 1, 'lw': 3}, add_cbar=False)

# Adjust layout for better spacing
plt.tight_layout()

# Calculate the cross pair correlation function (PCF) between populations A and B
r, PCF = ms.spatial_statistics.cross_pair_correlation_function(
    domain,
    population_A=('Collection', 'Macrophages'),
    population_B=('Collection', 'Macrophages'),
    max_R=200,
    annulus_step=5,
    annulus_width=10
)

# Plot the PCF
plt.figure(figsize=(8, 6))
plt.plot(r, PCF, lw=5)
plt.gca().axhline(1, c='k', linestyle=':', lw=3)
plt.ylim([0, 10])
plt.ylabel('$g_{AB}(r)$')
plt.xlabel('$r$')
plt.show()

# Calculate the cross pair correlation function (PCF) between populations A and B
# excluding the PanCK boundaries (stroma) and including only the PanCK boundaries (tumour)
r_stroma, PCF_stroma = ms.spatial_statistics.cross_pair_correlation_function(
    domain,
    population_A=('Collection', 'Macrophages'),
    population_B=('Collection', 'Macrophages'),
    exclude_boundaries=('collection', 'PanCK'),
    max_R=200,
    annulus_step=5,
    annulus_width=10
)
r_tumour, PCF_tumour = ms.spatial_statistics.cross_pair_correlation_function(
    domain,
    population_A=('Collection', 'Macrophages'),
    population_B=('Collection', 'Macrophages'),
    include_boundaries=('collection', 'PanCK'),
    max_R=200,
    annulus_step=5,
    annulus_width=10
)

# Plot the PCF for both stroma and tumour regions
plt.figure(figsize=(8, 6))
plt.plot(r, PCF, label='ROI', linestyle='--', lw=3)
plt.plot(r_stroma, PCF_stroma, label='stroma', lw=5, c=plt.cm.tab10(3))
plt.plot(r_tumour, PCF_tumour, label='tumour', lw=5, c=plt.cm.tab10(2))
plt.gca().axhline(1, c='k', linestyle=':', lw=3)
plt.ylim([0, 3])
plt.ylabel('$g(r)$')
plt.xlabel('$r$')
plt.legend()
plt.show()

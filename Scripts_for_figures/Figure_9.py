from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import addcopyfighandler
import contextily as ctx
import mplcursors as mpl
import os
import statsmodels.api as sm
from scipy.cluster import hierarchy
from scipy.spatial import distance

from ml_tools import clustering
from ml_tools import plots
from ml_tools import dist_calc
from openpyxl import load_workbook
matplotlib.rcParams['font.family'] = 'Times New Roman'
# Modify these as needed

# Minimum and maximum period to consider
# for clustering
# Set to None to use min/max of the data
min_period, max_period = None, None

# Window size for smoothing
# Averages (median) the specified number of
# data points into a single value
# Increasing this will reduce the effect of
# smaller peaks in the residual, see
# first plot for effect of this
use_smoothing = False
win_size = 5


# Specifies the Dynamic Time Warping
# radius constraint
DTW_radius = 0

# Linkage method
# Has to be one of:
#   complete, single, average, ward
# See here for details:
# https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
linkage_method = "ward"
RS = True

# Number of clusters
n_clusters = 4

def load_res_pSA_from_csv(im_data_ffp: Path):
    """Loads the residual PSA values from a csv file
    (I.e. Data from Ayushi)
    """
    im_df = pd.read_csv(im_data_ffp, index_col=0).T

    # Only interested in pSA
    pSA_columns = np.asarray([cur_col for cur_col in im_df.columns if cur_col.startswith("pSA")])
    im_df = im_df.loc[:, pSA_columns]

    # Get the pSA period values
    pSA_period_values = [float(cur_c.rsplit("_", maxsplit=1)[-1]) for cur_c in pSA_columns]
    im_df.columns = pSA_period_values

    # Ensure columns were already sorted
    assert np.all(np.sort(pSA_period_values) == pSA_period_values)

    # Drop any with nan
    im_df = im_df.dropna()

    return im_df
def save_to_excel(df, excel, sheet_name):
    with pd.ExcelWriter(excel, engine='openpyxl', mode='a') as writer:
        if sheet_name not in writer.book.sheetnames:
            df_clusters.to_excel(writer, sheet_name=f"Clusters = {n_clusters}, DTW = {DTW_radius}, smoothing = {use_smoothing}", index=False)
            
def hover_format(sel):
    index = sel.index
    if index is not None:
        name = df_site.loc[index, 'stat_name']
        cluster = df_site.loc[index, 'Cluster']
        Geom = df_site.loc[index, 'Geom']
        tooltip_text = f"{name}\n{Geom}\n{cluster}"
    sel.annotation.set(text=tooltip_text ,ha='center',va='center')

def on_add(sel):
    label = sel.artist.get_label()
    sel.annotation.set(text=label, ha='center', va='center')
    
label        = None
clicked_line = None
clicked_line_color = None
def onpick(event):
    global label, clicked_line, clicked_line_color
    if not isinstance(event.artist, plt.Line2D):
        return
    if label is not None:
        label.remove()
    if clicked_line is not None:
        clicked_line.set(color=clicked_line_color, zorder=2, linewidth=1)
    clicked_line = event.artist
    clicked_line_color = clicked_line.get_color()
    clicked_line.set(color='maroon', zorder=10, linewidth=5)
    plt.gcf().canvas.draw()
    
# Load the IM data
res_data_ffp = Path(
    r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\New Results\Residuals\PJSreStationBiased_sim.csv"
)
res_df = load_res_pSA_from_csv(res_data_ffp)
sim_sitereserr = load_res_pSA_from_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\New Results\Residuals\PJSreStationerror_sim.csv"))
emp_res_data_ffp = res_data_ffp = Path(
    r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Empirical residuals\v22p11p10\Residuals\PJSreStationBiased_emp.csv"
)
empres_df = load_res_pSA_from_csv(emp_res_data_ffp)
# FdS2S_df  = pd.read_csv(Path(r"C:\Users\User\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\FAS\PJSreStationBiased_sim.csv"),index_col='imName')
# FdS2S_df  = FdS2S_df.dropna(axis=1)
# FdS2S_df  = FdS2S_df.T
# Ffreq = pd.read_csv(Path(r"C:\Users\User\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\FAS\PJSvarCompsBiased_sim.csv"))
# Ffreq = Ffreq.iloc[:,1]

stations = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\4. Clustering\stations.csv"))
hillstatids       = stations[stations['Geom']=='Hill']['stat_id'].apply(lambda x: f'Station_{x}')
hillres_df        = res_df.loc[hillstatids]
hillerrres_df     = sim_sitereserr.loc[hillstatids]
X = hillres_df.copy()
variance = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\New Results\Residuals\PJSvarCompsBiased_sim.csv"))
varianceemp = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Empirical residuals\v22p11p10\Residuals\PJSvarCompsBiased_emp.csv"))
if RS:
    phiS2S   = variance.iloc[6:,4]
    phiS2S_emp = varianceemp.iloc[6:,4]
else:
    phiS2S   = variance.iloc[:,4]

    
# Filter periods
if min_period or max_period is not None:
    min_period = X.columns.min() if min_period is None else min_period
    max_period = X.columns.max() if max_period is None else max_period

    mask = (X.columns.values >= min_period) & (X.columns.values <= max_period)
    X = X.loc[:, mask]
    res_df = res_df.loc[:, mask]

# Determine needed shape
n_periods = int(np.ceil(X.shape[1] / win_size) * win_size)

# Smooth
if use_smoothing:
    X_smooth = np.pad(X, ((0, 0), (0, n_periods - X.shape[1])), "edge")
    period_values = np.median(
        np.pad(X.columns.values, (0, n_periods - X.shape[1]), "edge").reshape(-1, win_size),
        axis=1,
    )
    X_smooth = X_smooth.reshape((X.shape[0], -1, win_size)).mean(axis=2)
    X = pd.DataFrame(index=X.index, data=X_smooth, columns=period_values)



# Plot smoothed residual spectra
if use_smoothing:
    fig = plt.figure(figsize=(16, 10))
    color_palette = sns.color_palette("Paired", 10)

    for ix in range(10):
        plt.semilogx(
            res_df.columns, res_df.iloc[ix, :].values, linewidth=0.75, c=color_palette[ix]
        )
        plt.semilogx(
            period_values,
            X.iloc[ix, :],
            linewidth=0.75,
            c=color_palette[ix],
            linestyle="--",
        )
    plt.xlim(res_df.columns.min(), res_df.columns.max())
    plt.xlabel(f"Period")
    plt.ylabel(f"Residual - pSA")
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    plt.tight_layout()


# Compute the distance matrix
dist_matrix_df = dist_calc.compute_dtw_dist_matrix(
    X, dict(global_constraint="sakoe_chiba", sakoe_chiba_radius=DTW_radius)
)

# Run hierachical clustering
Z = clustering.compute_hierarchical_linkage_matrix(
    distance.squareform(dist_matrix_df.values), method=linkage_method
)
cluster_labels = hierarchy.fcluster(Z, n_clusters, criterion="maxclust")

# # Plot the dendrogram
# fig = plt.figure(figsize=(12, 9))
# dn = hierarchy.dendrogram(Z)
# plt.title(f"Dendrogram")
# plt.tight_layout()

# Getting the stations names and its geographical location in each cluster
stat_names = [None]*n_clusters
longitude  = [None]*n_clusters
latitude   = [None]*n_clusters
cluster_data = {}

for cluster_id in range(n_clusters):
    stats = hillres_df.index[cluster_labels==cluster_id+1]
    stats = [int(stations.rsplit("_",maxsplit=1)[-1]) for stations in stats]
    stat_names[cluster_id] = [str(stations.loc[i-1].stat_name) for i in stats]
    longitude[cluster_id] = [str(stations.loc[i-1].lon) for i in stats]
    latitude[cluster_id] = [str(stations.loc[i-1].lat) for i in stats]
    cluster_data[f'Cluster {cluster_id+1}'] = stat_names[cluster_id]
    
max_stations = max(len(stat_names) for stat_names in cluster_data.values())
df_clusters = pd.DataFrame({f'{cluster_id}': stat_names + [''] * (max_stations - len(stat_names)) for cluster_id, stat_names in cluster_data.items()})
filter_slope = pd.to_numeric(stations['Slope'], errors='coerce')
stations['Slope']  = filter_slope[~np.isnan(filter_slope)]
# df_clusters.to_excel('Clusters.xlsx', sheet_name = f"Clusters = {n_clusters}, DTW = {DTW_radius}",index=False)
# save_to_excel(df_clusters, 'Clusters.xlsx', sheet_name = f"Clusters = {n_clusters}, DTW = {DTW_radius}, smoothing = {use_smoothing}")
#%% Site characterization
data = []
lats = latitude
longs = longitude 
stats = stat_names
for cluster_label, (lats, longs, stats) in enumerate(zip(lats, longs,stats)):
    for lat, long, stats in zip(lats, longs,stats):
        data.append({'stat_name': stats,'lat': float(lat), 'lon': float(long), 'Cluster': cluster_label+1})
df_latlong = pd.DataFrame(data)
df_site = pd.merge(stations[['stat_id','stat_name','vs30','Geom','Z1_NZVM','Z1_Meas','T0','Slope','H250','H1250','r1250']],df_latlong,on='stat_name')
df_site['Ratio']=df_site['Z1_Meas']/df_site['Z1_NZVM']
df_site.replace([np.inf, -np.inf], np.nan, inplace=True)
df_site = df_site.sort_values(by='stat_name',ascending=True)
# culpritpath = r"C:\Users\User\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\2. Interpretations"
# df_culprits = pd.read_csv(os.path.join(culpritpath,'Culprits.txt'),sep='\t')
# df_culprits = pd.read_csv(os.path.join(culpritpath,'Culpritsgreaterhalf.txt'),sep='\t')
# df_culprits = df_culprits.rename(columns={'Station Code': 'stat_name'})
# df_culpritsite = pd.merge(df_culprits,df_site,on='stat_name')
hilldf_site    = df_site[df_site['Geom']=='Hill']
hillgrouped_data = hilldf_site.groupby('Cluster')
grouped_data = df_site.groupby('Cluster')
# grouped_dataproblematic = df_culpritsite.groupby('Cluster')
colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
num_rows = (n_clusters + 2) // 3 
num_cols = 3
folderpath= 'Culprits'
#%% Plot clusters - Paper 1
colors = ['red','blue','green','orange']
num_rows = (n_clusters + 2) // 3  
num_cols = 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10.1 ,  7.11),constrained_layout = True,sharex=True,sharey=True)

fig.text(0.0, 1, '(a)', fontsize=16, fontweight='bold', ha='left', va='top')
if num_rows == 1:
    axes = axes.reshape(1, -1)
elif num_cols == 1:
    axes = axes.reshape(-1, 1)

grouped_data = hillres_df.groupby(cluster_labels)
grouped_dataerr = hillerrres_df.groupby(cluster_labels)
cluster_mean = hillres_df.groupby(cluster_labels).mean()
cat_phiS2S = {}

for label, group_errdS2S in grouped_dataerr:
    group_dS2S = grouped_data.get_group(label)
    # Cat_std =  np.sqrt(np.mean((group_dS2S)**2,axis=0)+np.mean((group_errdS2S)**2,axis=0))
    Cat_std =  np.sqrt(np.mean((group_dS2S-np.mean(group_dS2S, axis=0))**2,axis=0)+np.mean((group_errdS2S)**2,axis=0))
    # Cat_std = group_dS2S.std(axis=0)
    cat_phiS2S[label]= Cat_std
cat_phiS2S  = pd.DataFrame(cat_phiS2S)
cat_phiS2S  = cat_phiS2S.transpose()

for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
    # Plot the cluster's data in the current subplot
    ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color='#a9a9a9', alpha=0.2,picker=True, pickradius=5, label=sorted(stat_names[cluster_id-1]))
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),colors[cluster_id-1],linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)-cat_phiS2S.iloc[cluster_id-1],colors[cluster_id-1],linestyle='--',linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)+cat_phiS2S.iloc[cluster_id-1],colors[cluster_id-1],linestyle='--',linewidth=2)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='k',linestyle='--')
    if (cluster_id-1) % num_cols == 0:
        ax.set_ylabel('${\it \delta S2S_s}$', size=18)
    ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
    # ax.set_title(f"Cluster {cluster_id}")
    ax.text(0.0105,1.2,f"Cluster {cluster_id}",fontsize=18, fontweight='bold')
    ax.text(0.02, -1.3, f"N = {len(cluster_data)}", fontweight='bold', fontsize=16)
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)

axes[0][0].text(0.3, 1.2, 'Underprediction', size=14, fontstyle='italic')
axes[0][0].text(0.35, -1.2, 'Overprediction', size=14, fontstyle='italic')
for ax in axes[1]:
    ax.set_xlabel('Vibration Period, T (s)', size=18)
fig.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)
for i in range(n_clusters, num_rows*num_cols):
    fig.delaxes(axes.flatten()[i])
# plt.savefig('Clusters_Hill.pdf')
#%%#%% One-panel version of Cluster Plot – Paper 1
fig, ax = plt.subplots(figsize=(11.05,  7.95),constrained_layout=True)
colors = ['red', 'blue', 'green', 'orange']

grouped_data = hillres_df.groupby(cluster_labels)
grouped_dataerr = hillerrres_df.groupby(cluster_labels)
cat_phiS2S = {}

for label, group_errdS2S in grouped_dataerr:
    group_dS2S = grouped_data.get_group(label)
    Cat_std = np.sqrt(
        np.mean((group_dS2S - np.mean(group_dS2S, axis=0))**2, axis=0) +
        np.mean((group_errdS2S)**2, axis=0)
    )
    cat_phiS2S[label] = Cat_std
cat_phiS2S = pd.DataFrame(cat_phiS2S).T  # shape: (n_clusters, n_T)

# Plot mean ± sigma for each cluster
for cluster_id in range(1, n_clusters + 1):
    cluster_data = grouped_data.get_group(cluster_id)
    mean_vals = cluster_data.values.T.mean(axis=1)
    std_vals = cat_phiS2S.iloc[cluster_id - 1].values
    periods = cluster_data.columns.astype(float)

    ax.semilogx(periods, mean_vals, color=colors[cluster_id - 1], linewidth=2, label=f'Cluster {cluster_id}')
    ax.fill_between(periods, mean_vals - std_vals, mean_vals + std_vals,
                    color=colors[cluster_id - 1], alpha=0.1)

ax.set_xlim([0.01, 10])
ax.set_ylim([-1.5, 1.5])
ax.axhline(0, color='k', linestyle='--')
ax.set_xlabel('Vibration Period, T (s)', size=35)
ax.set_ylabel('${\it \delta S2S_s}$', size=35)
ax.tick_params(labelsize=35, direction='in', which='both')
ax.grid(color='gray', linestyle='dashed', linewidth=0.4, which='both',alpha=0.5)
ax.legend(fontsize=26,frameon=False)
ax.text(0.011, 1.2, 'Underprediction', size=35, fontstyle='italic')
ax.text(0.011, -1.2, 'Overprediction', size=35, fontstyle='italic')
fig.text(0.0, 1, '(a)', fontsize=35, fontweight='bold', ha='left', va='top')
ax.tick_params(axis='x', pad=10)
plt.savefig('Clusters_Hill.pdf')

#%% Plot CDF of H1250 - Paper 1
selected_clusters = [1,2,3,4]
fig, ax = plt.subplots(figsize=(13.04,8.35),constrained_layout=True)

fig.text(0.0, 1, '(b)', fontsize=35, fontweight='bold', ha='left', va='top')
for i, cluster in enumerate(selected_clusters):
    group = hillgrouped_data.get_group(cluster)
    total = grouped_data.get_group(cluster)
    ecdf = sm.distributions.ECDF(group['H1250'])
    x = np.linspace(min(group['H1250']), max(group['H1250']), 1000)
    y = ecdf(x)
    ax.plot(x, y, label=f'Cluster {i+1}', color=colors[i % len(colors)], linestyle='-',linewidth=4)

ax.set_xlabel('$\mathregular{H_{1250}\ (m)}$', size=35)
ax.set_ylabel('Cumulative distribution function', size=35)
ax.set_xlim([-100,100])
ax.tick_params(labelbottom=True, labelleft=True,labelsize=35)
# ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
# ax.legend(fontsize=30,frameon=False)
plt.savefig('Hills_H1250CDFs.pdf')
#%% Roughness
selected_clusters = [1,2,3,4]
fig, ax = plt.subplots(figsize=(13.04,8.35),constrained_layout=True)

fig.text(0.0, 1, '(c)', fontsize=35, fontweight='bold', ha='left', va='top')
for i, cluster in enumerate(selected_clusters):
    group = hillgrouped_data.get_group(cluster)
    total = grouped_data.get_group(cluster)
    ecdf = sm.distributions.ECDF(group['r1250'])
    x = np.linspace(min(group['r1250']), max(group['r1250']), 1000)
    y = ecdf(x)
    ax.plot(x, y, label=f'Cluster {i+1}', color=colors[i % len(colors)], linestyle='-',linewidth=4)

ax.set_xlabel('$\mathregular{r_{1250}\ (m)}$', size=35)
ax.set_ylabel('Cumulative distribution function', size=35)
# ax.set_xlim([-100,100])
ax.tick_params(labelbottom=True, labelleft=True,labelsize=35)
# ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
# ax.legend(fontsize=35,frameon=False)
plt.savefig('Hills_r1250CDFs.pdf')
#%% Plot CDF of vs30 - Paper 1
selected_clusters = [1,2,3,4]
fig, ax = plt.subplots(figsize=(13.04,8.35),constrained_layout=True)

fig.text(0.0, 1, '(a)', fontsize=35, fontweight='bold', ha='left', va='top')
for i, cluster in enumerate(selected_clusters):
    group = hillgrouped_data.get_group(cluster)
    total = grouped_data.get_group(cluster)
    ecdf = sm.distributions.ECDF(group['vs30'])
    x = np.linspace(min(group['vs30']), max(group['vs30']), 1000)
    y = ecdf(x)
    ax.plot(x, y, label=f'Cluster {i+1}', color=colors[i % len(colors)], linestyle='-',linewidth=4)

ax.set_xlabel('$\mathregular{V_{S30}\ (m/s)}$', size=35)
ax.set_ylabel('Cumulative distribution function', size=35)
# ax.set_xlim([-85,85])
ax.tick_params(labelbottom=True, labelleft=True,labelsize=35)
# ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.legend(fontsize=35,frameon=False)
plt.savefig('Hills_Vs30CDFs.pdf')
#%% Plot CDF of T0 - Paper 1
selected_clusters = [1,2,3,4]
fig, ax = plt.subplots(figsize=(13.04,8.35),constrained_layout=True)

fig.text(0.0, 1, '(b)', fontsize=35, fontweight='bold', ha='left', va='top')
for i, cluster in enumerate(selected_clusters):
    group = hillgrouped_data.get_group(cluster)
    total = grouped_data.get_group(cluster)
    ecdf = sm.distributions.ECDF(group['T0'])
    x = np.linspace(min(group['T0']), max(group['T0']), 1000)
    y = ecdf(x)
    ax.plot(x, y, label=f'Cluster {i+1}', color=colors[i % len(colors)], linestyle='-',linewidth=4)

ax.set_xlabel('$\mathregular{T_{0}\ (s)}$', size=35)
ax.set_ylabel('Cumulative distribution function', size=35)
# ax.set_xlim([-85,85])
ax.tick_params(labelbottom=True, labelleft=True,labelsize=35)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.legend(fontsize=35,frameon=False)
plt.savefig('Hills_T0CDFs.pdf')
#%% Plot CDF of Z1_Meas - Paper 1
selected_clusters = [1,2,3,4]
fig, ax = plt.subplots(figsize=(13.04,8.35),constrained_layout=True)

fig.text(0.0, 1, '(b)', fontsize=35, fontweight='bold', ha='left', va='top')
for i, cluster in enumerate(selected_clusters):
    group = hillgrouped_data.get_group(cluster)
    total = grouped_data.get_group(cluster)
    ecdf = sm.distributions.ECDF(group['Z1_Meas'])
    x = np.linspace(min(group['Z1_Meas']), max(group['Z1_Meas']), 1000)
    y = ecdf(x)
    ax.plot(x, y, label=f'Cluster {i+1}', color=colors[i % len(colors)], linestyle='-',linewidth=4)

ax.set_xlabel('$\mathregular{Z_{1.0}\ (m)}$', size=35)
ax.set_ylabel('Cumulative distribution function', size=35,color='white')
# ax.set_xlim([-85,85])
ax.tick_params(labelbottom=True, labelleft=True,labelsize=35)
# ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
# ax.legend(fontsize=35,frameon=False)
plt.savefig('Hills_Z1_MeasCDFs.pdf')
#%% Plot CDF of Slope - Paper 1
selected_clusters = [1,2,3,4]
fig, ax = plt.subplots(figsize=(13.04,8.35),constrained_layout=True)

fig.text(0.0, 1, '(d)', fontsize=35, fontweight='bold', ha='left', va='top')
for i, cluster in enumerate(selected_clusters):
    group = hillgrouped_data.get_group(cluster)
    total = grouped_data.get_group(cluster)
    ecdf = sm.distributions.ECDF(group['Slope'])
    x = np.linspace(min(group['Slope']), max(group['Slope']), 1000)
    y = ecdf(x)
    ax.plot(x, y, label=f'Cluster {i+1}', color=colors[i % len(colors)], linestyle='-',linewidth=4)

ax.set_xlabel('Topographic slope (m/m)', size=35)
ax.set_ylabel('Cumulative distribution function', size=35,color='white')
# ax.set_xlim([-85,85])
ax.tick_params(labelbottom=True, labelleft=True,labelsize=35)
# ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
# ax.legend(fontsize=35,frameon=False)
plt.savefig('Hills_SlopeCDFs.pdf')
#%% Plot clusters - QC Poster
# Define the desired cluster indices
selected_clusters = [5,1,4,6]
colorsnew = [colors[7],colors[2],'b','g']
grouped_data = res_df.groupby(cluster_labels)
# Calculate the number of rows and columns for subplots
num_selected_clusters = len(selected_clusters)
num_rows = (num_selected_clusters + 2) // 3
num_cols = 2

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18.22,9.86),constrained_layout=True,sharex=True,sharey=True)
if num_rows == 1:
    axes = axes.reshape(1, -1)
elif num_cols == 1:
    axes = axes.reshape(-1, 1)


# Loop through the selected clusters and their indices
for i, cluster_id in enumerate(selected_clusters):
    cluster_data = grouped_data.get_group(cluster_id)
    ax = axes.flatten()[i]

    # Plot the cluster's data in the current subplot
    ax.semilogx(cluster_data.columns, cluster_data.values.T, linewidth=1, color=colorsnew[i])
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1), 'k', linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1) - cluster_data.values.T.std(axis=1), 'k--', linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1) + cluster_data.values.T.std(axis=1), 'k--', linewidth=2)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0, color='maroon', linestyle='-')
    if i % num_cols == 0:
        ax.set_ylabel(r'$\mathregular{\delta S2S_s}$', size=50)
        # ax.set_ylabel('${\it \delta S2S_s}$', size=30)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.tick_params(labelsize=24)
    ax.text(1.75,1.2,f"Cluster {i + 1}",fontsize=35, fontweight='bold')
    ax.text(1.3, -1.05, '$\mathregular{N_{sites}}$ = %d'%len(cluster_data),fontsize=40)
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes[0][0].text(0.013, 1.2, 'Underprediction', size=35)
axes[0][0].text(0.013, -1.39, 'Overprediction', size=35)
for ax in axes[1]:
    ax.set_xlabel('Vibration Period, T (s)', size=40)
#%% Plot Histograms - Geomorphology QC Poster
selected_clusters = [5,1,4,6]
colorsnew = [colors[7],colors[2],'b','g']
grouped_data = df_site.groupby('Cluster')
bin_labels = np.array(['Hill','Valley','Basin-edge','Basin'])

bin_percentages = []
for cluster_id in selected_clusters:
    group = grouped_data.get_group(cluster_id)
    total_sites = len(group)
    cluster_counts = np.array([np.sum(group['Geom'] == bin_name) for bin_name in bin_labels])
    cluster_percentages = cluster_counts / total_sites
    bin_percentages.append(cluster_percentages)

bin_percentages = np.array(bin_percentages)

bar_width = 1
fig, ax = plt.subplots(figsize=(9.59,6.48),constrained_layout=True)
x_positions = np.arange(len(bin_labels)) * (len(selected_clusters) + 1)
for i, bin_label in enumerate(bin_labels):
    cluster_offsets = np.arange(len(selected_clusters)) * bar_width
    colors_for_bin = [colorsnew[i] for _ in selected_clusters]  # Get colors for each cluster
    ax.bar(x_positions + cluster_offsets, bin_percentages[i, :], bar_width, label=f"Cluster {i+1}", color=colors_for_bin)
    x_positions = x_positions.astype(float) + bar_width  # Ensure float addition
    # colors_for_bin = [colors[j % len(colors)] for j in selected_clusters]
    # ax.bar(x_positions, bin_percentages[:, i], bar_width, color=colors_for_bin)
ax.set_xlabel('Geomorphic category', size=40)
ax.set_ylabel('% sites', size=40)
ax.tick_params(labelsize=24)
ax.legend(fontsize=30,frameon=False)
ax.set_xticks(np.arange(len(bin_labels)) * (len(selected_clusters) + 2) + (len(selected_clusters) - 1) / 2)
ax.set_xticklabels(bin_labels)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
#%% Selected clusters - Vs30 dependency - All
selected_clusters = [5,1,4,6]
colorsnew = [colors[7],colors[2],'b','g']
fig, axes = plt.subplots(2, 2, figsize=(8.77, 5.53),constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
logbinsVs30 = np.linspace(min(df_site['vs30']), max(df_site['vs30']), 10)
for i,cluster in enumerate(selected_clusters):
    ax = axes[i]
    group = grouped_data.get_group(cluster)
    ax.hist(group['vs30'],bins=logbinsVs30,weights=np.ones(len(group['vs30'])) / len(group['vs30']),color=colorsnew[i % len(colors)], edgecolor='k')
    ax.text(1000,0.45,f'Cluster {i+1}',fontsize=16, fontweight='bold')
    if i % 2 == 0:
        ax.set_ylabel('% sites', size=16)
    mean = group['vs30'].mean()
    std_dev = group['vs30'].std()
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    for ax in axes[-2:]:
        ax.set_xlabel('$V_{S30} (m/s)$', size=16)
#%% Selected clusters - Vs30 dependency - Only hills
selected_clusters = [5,1,4,6]
colorsnew = [colors[7],colors[2],'b','g']
fig, axes = plt.subplots(2, 2, figsize = (16.34,9.74),constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
hilldf_sitesel = hilldf_site[hilldf_site['Cluster'].isin(selected_clusters)]
logbinsVs30 = np.linspace(min(hilldf_sitesel['vs30']), max(hilldf_sitesel['vs30']), 7)
for i,cluster in enumerate(selected_clusters):
    ax = axes[i]
    total = grouped_data.get_group(cluster)
    group = hillgrouped_data.get_group(cluster)
    ax.hist(group['vs30'],bins=logbinsVs30,weights=np.ones(len(group['vs30'])) / len(total['vs30']),color=colorsnew[i % len(colors)], edgecolor='k')
    ax.text(800,0.185,f'Cluster {i+1}',fontsize=40, fontweight='bold')
    if i % 2 == 0:
        ax.set_ylabel('% hill sites', size=40)
    mean = group['vs30'].mean()
    std_dev = group['vs30'].std()
    ax.tick_params(labelbottom=True, labelleft = True,labelsize=24)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    for ax in axes[-2:]:
        ax.set_xlabel('$\mathregular{V_{S30}\ (m/s)}$', size=40)
#%% Selected clusters - T0 dependency - All
def custom_formatter(x, pos):
    if x == int(x):
        return "{:.0f}".format(x)  # Display whole numbers without decimals
    else:
        return "{:.2f}".format(x)
selected_clusters = [5,1,4,6]
colorsnew = [colors[7],colors[2],'b','g']
fig, axes = plt.subplots(2, 2, figsize=(8.77, 5.53),constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
logbinsT0 = np.logspace(np.log10(df_site['T0'].min()),np.log10(df_site['T0'].max()),7)
for i,cluster in enumerate(selected_clusters):
    ax = axes[i]
    group = grouped_data.get_group(cluster)
    ax.hist(group['T0'],bins=logbinsT0,weights=np.ones(len(group['T0'])) / len(group['T0']),color=colorsnew[i % len(colors)], edgecolor='k')
    ax.text(1.5,0.45,f'Cluster {i+1}',fontsize=16, fontweight='bold')
    if i % 2 == 0:
        ax.set_ylabel('% sites', size=16)
    mean = group['T0'].mean()
    std_dev = group['T0'].std()
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.set_xscale('log')
    ax.set_xticks([0.05,0.1,0.5,1,2,3,6])    
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(custom_formatter))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    for ax in axes[-2:]:
        ax.set_xlabel('$T_0\ (s)$', size=16)
#%% Selected clusters - T0 dependency - Only hills
selected_clusters = [5,1,4,6]
colorsnew = [colors[7],colors[2],'b','g']
fig, axes = plt.subplots(2, 2, figsize = (16.34,9.74),constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
hilldf_sitesel = hilldf_site[hilldf_site['Cluster'].isin(selected_clusters)]
logbinsT0 = np.logspace(np.log10(hilldf_sitesel['T0'].min()),np.log10(hilldf_sitesel['T0'].max()),6)
for i,cluster in enumerate(selected_clusters):
    ax = axes[i]
    total = grouped_data.get_group(cluster)
    group = hillgrouped_data.get_group(cluster)
    ax.hist(group['T0'],bins=logbinsT0,weights=np.ones(len(group['T0'])) / len(total['T0']),color=colorsnew[i % len(colors)], edgecolor='k')
    ax.text(0.15,0.45,f'Cluster {i+1}',fontsize=40, fontweight='bold')
    if i % 2 == 0:
        ax.set_ylabel('% hill sites', size=40)
    ax.tick_params(labelbottom=True, labelleft = True,labelsize=24)
    ax.set_xscale('log')    
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(custom_formatter))
    ax.set_xticks([0.05,0.1,0.2,0.3])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
for ax in axes[-2:]:
    ax.set_xlabel('$T_0\ (s)$', size=40)
#%% Selected clusters - H250 dependency - All
selected_clusters = [5,1,4,6]
colorsnew = [colors[7],colors[2],'b','g']
fig, axes = plt.subplots(2, 2, figsize=(8.77, 5.53),constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
logbinsH250 = [-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21,24,25]
for i,cluster in enumerate(selected_clusters):
    ax = axes[i]
    group = grouped_data.get_group(cluster)
    ax.hist(group['H250'],bins=logbinsH250,weights=np.ones(len(group['H250'])) / len(group['H250']),color=colorsnew[i % len(colors)], edgecolor='k')
    ax.text(12,0.4,f'Cluster {i+1}',fontsize=16, fontweight='bold')
    if i % 2 == 0:
        ax.set_ylabel('% sites', size=16)
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    for ax in axes[-2:]:
        ax.set_xlabel('$H_{250} (m)$', size=16)
#%% Selected clusters - H250 dependency - Only hills
selected_clusters = [5,1,4,6]
colorsnew = [colors[7],colors[2],'b','g']
fig, axes = plt.subplots(2, 2,constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
hilldf_sitesel = hilldf_site[hilldf_site['Cluster'].isin(selected_clusters)]
logbinsH250 = np.linspace(min(hilldf_sitesel['H250']), max(hilldf_sitesel['H250']), 6)
for i,cluster in enumerate(selected_clusters):
    ax = axes[i]
    total = grouped_data.get_group(cluster)
    group = hillgrouped_data.get_group(cluster)
    ax.hist(group['H250'],bins=logbinsH250,weights=np.ones(len(group['H250'])) / len(total['H250']),color=colorsnew[i % len(colors)], edgecolor='k')
    ax.text(6,0.37,f'Cluster {i+1}',fontsize=16, fontweight='bold')
    if i % 2 == 0:
        ax.set_ylabel('% sites', size=16)
    ax.set_xticks([-15,-6,1,9,18,25]) 
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    for ax in axes[-2:]:
        ax.set_xlabel('$H_{250} (m)$', size=16)
#%% CDF of hill sites
selected_clusters = [5, 1,4,6]
colorsnew = [colors[7], colors[2], 'b', 'g']
fig, ax = plt.subplots(figsize=(13.04,8.35),constrained_layout=True)

for i, cluster in enumerate(selected_clusters):
    group = hillgrouped_data.get_group(cluster)
    total = grouped_data.get_group(cluster)
    ecdf = sm.distributions.ECDF(group['H250'])
    x = np.linspace(min(group['H250']), max(group['H250']), 1000)
    y = ecdf(x)
    ax.plot(x, y, label=f'Cluster {i+1}', color=colorsnew[i % len(colors)], linestyle='-',linewidth=5)

ax.set_xlabel('$\mathregular{H_{250}\ (m)}$', size=40)
ax.set_ylabel('CDF of hill sites', size=40)
# ax.set_xlim([-25,25])
ax.tick_params(labelbottom=True, labelleft=True,labelsize=30)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.legend(fontsize=35,frameon=False)
#%% CDF of hill sites - H1250
selected_clusters = [5,1,4,6]
colorsnew = [colors[7], colors[2], 'b', 'g']
fig, ax = plt.subplots(figsize=(13.04,8.35),constrained_layout=True)

for i, cluster in enumerate(selected_clusters):
    group = hillgrouped_data.get_group(cluster)
    total = grouped_data.get_group(cluster)
    ecdf = sm.distributions.ECDF(group['H1250'])
    x = np.linspace(min(group['H1250']), max(group['H1250']), 1000)
    y = ecdf(x)
    ax.plot(x, y, label=f'Cluster {i+1}', color=colorsnew[i % len(colors)], linestyle='-',linewidth=5)

ax.set_xlabel('$\mathregular{H_{1250}\ (m)}$', size=40)
ax.set_ylabel('CDF of hill sites', size=40)
# ax.set_xlim([-25,25])
ax.tick_params(labelbottom=True, labelleft=True,labelsize=30)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.legend(fontsize=35,frameon=False)
#%% Selected clusters - Roughness dependency - All
selected_clusters = [5,1,4,6]
colorsnew = [colors[7],colors[2],'b','g']
fig, axes = plt.subplots(2, 2, figsize=(8.77, 5.53),constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
logbinsr370 = np.logspace(np.log10(df_site['r370'].min()),np.log10(df_site['r370'].max()),10)
for i,cluster in enumerate(selected_clusters):
    ax = axes[i]
    group = grouped_data.get_group(cluster)
    ax.hist(group['r370'],bins=logbinsr370,weights=np.ones(len(group['r370'])) / len(group['r370']),color=colorsnew[i % len(colors)], edgecolor='k')
    ax.text(80,0.27,f'Cluster {i+1}',fontsize=16, fontweight='bold')
    if i % 2 == 0:
        ax.set_ylabel('% sites', size=16)
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.set_xscale('log')
    ax.set_xticks([10,20,30,40,50,100,200])  
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    for ax in axes[-2:]:
        ax.set_xlabel('Roughness (r370)', size = 16)
#%% Selected clusters - Roughness dependency - Only hills
selected_clusters = [5,1,4,6]
colorsnew = [colors[7],colors[2],'b','g']
fig, axes = plt.subplots(2, 2, figsize = (7.5,5.58),constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
hilldf_sitesel = hilldf_site[hilldf_site['Cluster'].isin(selected_clusters)]
logbinsr370 = np.logspace(np.log10(hilldf_sitesel['r370'].min()),np.log10(hilldf_sitesel['r370'].max()),5)
for i,cluster in enumerate(selected_clusters):
    ax = axes[i]
    total = grouped_data.get_group(cluster)
    group = hillgrouped_data.get_group(cluster)
    ax.hist(group['r370'],bins=logbinsr370,weights=np.ones(len(group['r370'])) / len(total['r370']),color=colorsnew[i % len(colors)], edgecolor='k')
    ax.text(30,0.3,f'Cluster {i+1}',fontsize=16, fontweight='bold')
    if i % 2 == 0:
        ax.set_ylabel('% sites', size=16)
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.set_xscale('log')
    ax.set_xticks([25,50,80,140,250])  
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    for ax in axes[-2:]:
        ax.set_xlabel('Roughness', size = 16)
#%% Selected clusters - Slope dependency - All
selected_clusters = [5,1,4,6]
colorsnew = [colors[7],colors[2],'b','g']
fig, axes = plt.subplots(2, 2, figsize=(8.77, 5.53),constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
logbinsSlope = np.linspace(min(df_site['Slope']), max(df_site['Slope']), 11)
for i,cluster in enumerate(selected_clusters):
    ax = axes[i]
    group = grouped_data.get_group(cluster)
    ax.hist(group['Slope'],bins=logbinsSlope,weights=np.ones(len(group['Slope'])) / len(group['Slope']),color=colorsnew[i % len(colors)], edgecolor='k')
    ax.text(0.2,0.5,f'Cluster {i+1}',fontsize=16, fontweight='bold')
    if i % 2 == 0:
        ax.set_ylabel('% sites', size=16)
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    for ax in axes[-2:]:
        ax.set_xlabel('Slope (m/m)', size = 16)
#%% Selected clusters - Slope dependency - Only hills
selected_clusters = [5,1,4,6]
colorsnew = [colors[7],colors[2],'b','g']
fig, axes = plt.subplots(2, 2, figsize = (7.5,5.58),constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
hilldf_sitesel = hilldf_site[hilldf_site['Cluster'].isin(selected_clusters)]
logbinsSlope = np.linspace(min(hilldf_sitesel['Slope']), max(hilldf_sitesel['Slope']), 7)
for i,cluster in enumerate(selected_clusters):
    ax = axes[i]
    total = grouped_data.get_group(cluster)
    group = hillgrouped_data.get_group(cluster)
    ax.hist(group['Slope'],bins=logbinsSlope,weights=np.ones(len(group['Slope'])) / len(total['Slope']),color=colorsnew[i % len(colors)], edgecolor='k')
    ax.text(0.2,0.25,f'Cluster {i+1}',fontsize=16, fontweight='bold')
    if i % 2 == 0:
        ax.set_ylabel('% sites', size=16)
    mean = group['Slope'].mean()
    std_dev = group['Slope'].std()
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    for ax in axes[-2:]:
        ax.set_xlabel('Slope (m/m)', size = 16)
# %%All sites
# Z0 - NZVM vs Vs30
fig30,ax10  = plt.subplots(figsize=(19.2,9.83),constrained_layout=True)
scatter1  = ax10.scatter(df_site['Z1_NZVM'],df_site['vs30'],label='',c='w')
mpl.cursor(scatter1,multiple=True).connect("add", hover_format)
for i, (name, group) in enumerate(grouped_data):
    ax10.scatter(group['Z1_NZVM'], group['vs30'],label=f"Cluster {name}", color=colors[i % len(colors)], edgecolor='k', s=50)
ax10.set_xscale('log')
ax10.set_xlabel('$Z_{1.0}(m) - NZVM $', size=16)
ax10.set_ylabel('$V_{S30} (m/s)$', size=16)
ax10.legend(fontsize="16", frameon=False)
# Z0 - Liam's Database vs Vs30
fig31,ax11  = plt.subplots(figsize=(19.2,9.83),constrained_layout=True)
scatter2  = ax11.scatter(df_site['Z1_Meas'],df_site['vs30'],label='',c='w')
mpl.cursor(scatter2,multiple=True).connect("add", hover_format)
for i, (name, group) in enumerate(grouped_data):
    ax11.scatter(group['Z1_Meas'], group['vs30'],label=f"Cluster {name}", color=colors[i % len(colors)], edgecolor='k', s=50)
ax11.set_xscale('log')
ax11.set_xlabel('$Z_{1.0}(m)- database$', size=16)
ax11.set_ylabel('$V_{S30} (m/s)$', size=16)
ax11.legend(fontsize="16", frameon=False)
fig32,ax12  = plt.subplots(figsize=(19.2,9.83),constrained_layout=True)
scatter2  = ax12.scatter(df_site['T0'],df_site['vs30'],label='',c='w')
mpl.cursor(scatter2,multiple=True).connect("add", hover_format)
for i, (name, group) in enumerate(grouped_data):
    ax12.scatter(group['T0'], group['vs30'],label=f"Cluster {name}", color=colors[i % len(colors)], edgecolor='k', s=50)
ax12.set_xscale('log')
ax12.set_xlabel('$T_0\  (s)$', size=16)
ax12.set_ylabel('$V_{S30} (m/s)$', size=16)
ax12.legend(fontsize="16", frameon=False)
#%% Histograms
# Vs30 for each cluster
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
logspace = False
if logspace:
    logbinsVs30 = np.logspace(np.log10(df_site['vs30'].min()),np.log10(df_site['vs30'].max()),7)
else:
    logbinsVs30 = np.linspace(min(df_site['vs30']), max(df_site['vs30']), 10)
for i, (name, group) in enumerate(grouped_data):
    ax = axes[i]
    ax.hist(group['vs30'],bins = logbinsVs30,weights = np.ones(len(group['vs30'])) / len(group['vs30']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size=16)
    mean = group['vs30'].mean()
    std_dev = group['vs30'].std()
    ax.tick_params(labelbottom=True, labelleft = True)
    if logspace:
        ax.set_xscale('log')
        ax.set_xticks([200,400,600,800,1000])    
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    # ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    # ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes[-num_cols:]:
    ax.set_xlabel('$V_{S30} (m/s)$', size=16)
# fig.savefig(os.path.join('Histograms',f'Vs30 Cluster Histogram_{logspace}.png'))
#%% T0 for each cluster
fig3, axes3 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
axes3 = axes3.flatten()
logspace = True
if logspace:
    logbinsT0 = np.logspace(np.log10(df_site['T0'].min()),np.log10(df_site['T0'].max()),7)
else:
    logbinsT0 = np.linspace(min(df_site['T0']), max(df_site['T0']), 10)
for i, (name, group) in enumerate(grouped_data):
    ax = axes3[i]
    ax.hist(group['T0'],bins = logbinsT0, weights = np.ones(len(group['T0'])) / len(group['T0']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size=16)
    mean = group['T0'].mean()
    std_dev = group['T0'].std()
    ax.tick_params(labelbottom=True, labelleft = True)
    if logspace:
        ax.set_xscale('log')
        ax.set_xticks([0.05,0.1,0.5,1,2,3,6])    
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    # ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    # ax.annotate(f'Mean: {mean:.1f}\nStd: {std_dev:.1f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes[-num_cols:]:
    ax.set_xlabel('$T_0 (s)$', size = 16)
# fig3.savefig(os.path.join('Histograms',f'T0 Cluster Histogram_{logspace}.png'))
#%% Geomorphology for each cluster
fig3, axes3 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
axes3 = axes3.flatten()
bins = np.array(['Basin','Basin-edge','Valley','Hill'])
for i, (name, group) in enumerate(grouped_data):
    ax = axes3[i]
    counts = np.array([np.sum(group == bin_name) for bin_name in bins])
    percentage  = counts/np.sum(counts)
    percentage  = np.sum(percentage,axis=1)
    ax.bar(bins, percentage, color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    ax.tick_params(labelbottom=True, labelleft = True)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
for ax in axes3[-num_cols:]:
    ax.set_xlabel('Geomorphology', size = 16)
fig3.savefig(os.path.join('Histograms','Geomorphology Cluster Histogram.png'))   
#%% Slope for each cluster
fig3, axes3 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
axes3 = axes3.flatten()
logspace = False
if logspace:
    logbinsslope = np.logspace(np.log10(df_site['Slope'].min()),np.log10(df_site['Slope'].max()),7)
else:
    logbinsslope = np.linspace(min(df_site['Slope']), max(df_site['Slope']), 11)
# logbinsslope = [0,0.01099,0.03099,0.05099,0.07099,0.10099,0.12099,0.15099,0.17099,0.20099,0.22099,0.25099,0.27099,0.30099,0.32]
for i, (name, group) in enumerate(grouped_data):
    ax = axes3[i]
    ax.hist(group['Slope'], bins=logbinsslope, weights = np.ones(len(group['Slope'])) / len(group['Slope']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    mean = group['Slope'].mean()
    std_dev = group['Slope'].std()
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    ax.tick_params(labelbottom=True, labelleft = True)
    if logspace:
        ax.set_xscale('log')
        # ax.set_xticks([0.05,0.1,0.5,1,2,3,6])    
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    ax.annotate(f'Mean: {mean:.2f}\nStd: {std_dev:.2f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
    for ax in axes3[-num_cols:]:
        ax.set_xlabel('Slope (m/m)', size = 16)
fig3.savefig(os.path.join('Histograms','Slope Cluster Histogram.png')) 
#%% Z 1.0 for each cluster
fig4, axes4 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
logbinsZ1 =[0,10,20,30,50,100,200,500,1000,1500]
axes4 = axes4.flatten()
for i, (name, group) in enumerate(grouped_data):
    ax = axes4[i]
    ax.hist(group['Z1_NZVM'], bins=logbinsZ1, weights = np.ones(len(group['Z1_NZVM'])) / len(group['Z1_NZVM']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    mean = group['Z1_NZVM'].mean()
    std_dev = group['Z1_NZVM'].std()
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
    ax.set_xscale('log')
    ax.set_xticks([10,20,30,50,100,200,500,1000,1500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
for ax in axes4[-num_cols:]:
    ax.set_xlabel('Z1_NZVM', size = 16)
fig4.savefig(os.path.join('Histograms','Z1 NZVM Cluster Histogram.png'))
#%% Z 1.0 measured for each cluster
fig5, axes5 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
logbinsZ1 =[0,10,20,30,50,100,200,500,1000,1500]
axes5 = axes5.flatten()
for i, (name, group) in enumerate(grouped_data):
    ax = axes5[i]
    ax.hist(group['Z1_Meas'], bins=logbinsZ1, weights = np.ones(len(group['Z1_Meas'])) / len(group['Z1_Meas']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    mean = group['Z1_Meas'].mean()
    std_dev = group['Z1_Meas'].std()
    ax.set_xscale('log')
    ax.set_xticks([10,20,30,50,100,200,500,1000,1500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes5[-num_cols:]:
    ax.set_xlabel('Z1_database', size = 16)
fig5.savefig(os.path.join('Histograms','Z1 database Cluster Histogram.png'))
#%% Ratio Z 1.0 for each cluster
fig6, axes6 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
axes6 = axes6.flatten()
logbinsratio = [0,0.1,0.3,0.5,0.8,1,1.2,1.6,2,4,8,10,12]
for i, (name, group) in enumerate(grouped_data):
    ax = axes6[i]
    ax.hist(group['Ratio'], bins=logbinsratio, weights = np.ones(len(group['Ratio'])) / len(group['Ratio']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    mean = group['Ratio'].mean()
    std_dev = group['Ratio'].std()
    ax.set_xscale('log')
    ax.set_xticks([0.1,0.3,0.5,0.8,1.2,2,4,8,12])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    # ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes6[-num_cols:]:
    ax.set_xlabel('Ratio (Z1_database/Z1_NZVM)', size = 16)
fig6.savefig(os.path.join('Histograms','Z1 Ratio Cluster Histogram.png'))
#%% Roughness for each cluster (r370)
fig6, axes6 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
axes6 = axes6.flatten()
logspace = True
if logspace:
    logbinsr370 = np.logspace(np.log10(df_site['r370'].min()),np.log10(df_site['r370'].max()),10)
else:
    logbinsr370 = np.linspace(min(df_site['r370']), max(df_site['r370']), 11)
for i, (name, group) in enumerate(grouped_data):
    ax = axes6[i]
    ax.hist(group['r370'], bins=logbinsr370, weights = np.ones(len(group['r370'])) / len(group['r370']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    mean = group['r370'].mean()
    std_dev = group['r370'].std()
    ax.set_xscale('log')
    ax.set_xticks([10,20,30,40,50,100,200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    # ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes6[-num_cols:]:
    ax.set_xlabel('Roughness (r370)', size = 16)
fig6.savefig(os.path.join('Histograms','Roughness Cluster Histogram.png'))
fig7, axes7 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
axes7 = axes7.flatten()
logspace = True
if logspace:
    logbinsr1260 = np.logspace(np.log10(df_site['r1260'].min()),np.log10(df_site['r1260'].max()),10)
else:
    logbinsr1260 = np.linspace(min(df_site['r1260']), max(df_site['r1260']), 11)
for i, (name, group) in enumerate(grouped_data):
    ax = axes7[i]
    ax.hist(group['r1260'], bins=logbinsr1260, weights = np.ones(len(group['r1260'])) / len(group['r1260']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    mean = group['r1260'].mean()
    std_dev = group['r1260'].std()
    ax.set_xscale('log')
    ax.set_xticks([10,20,30,40,50,100,200,500,1000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    # ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes7[-num_cols:]:
    ax.set_xlabel('Roughness (r1260)', size = 16)
fig7.savefig(os.path.join('Histograms','Roughness 1260 Cluster Histogram.png'))
#%% H250 for each cluster
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
logbinsH250 = [-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21,24,25]
for i, (name, group) in enumerate(grouped_data):
    ax = axes[i]
    ax.hist(group['H250'],bins = logbinsH250,weights = np.ones(len(group['H250'])) / len(group['H250']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size=16)
    mean = group['H250'].mean()
    std_dev = group['H250'].std()
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    # ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    # ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes[-num_cols:]:
    ax.set_xlabel('$H_{250} (m)$', size=16)
fig.savefig(os.path.join('Histograms',f'H250 Cluster Histogram.png'))
#%% H1250 for each cluster
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True, sharex=True, sharey=True)
axes = axes.flatten()
logbinsH1250 = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]
for i, (name, group) in enumerate(grouped_data):
    ax = axes[i]
    ax.hist(group['H1250'],bins = logbinsH1250,weights = np.ones(len(group['H1250'])) / len(group['H1250']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size=16)
    mean = group['H1250'].mean()
    std_dev = group['H1250'].std()
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    # ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    # ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes[-num_cols:]:
    ax.set_xlabel('$H_{1250} (m)$', size=16)
fig.savefig(os.path.join('Histograms',f'H1250 Cluster Histogram.png'))
#%% Problematic sites
# Z0 - NZVM vs Vs30
fig33,ax13  = plt.subplots(figsize=(19.2,9.83),constrained_layout=True)
scatter4  = ax13.scatter(df_site['Z1_NZVM'],df_site['vs30'],label='',c='w')
mpl.cursor(scatter4,multiple=True).connect("add", hover_format)
for i, (name, group) in enumerate(grouped_dataproblematic):
    ax13.scatter(group['Z1_NZVM'], group['vs30'],label=f"Cluster {name}", color=colors[i % len(colors)], edgecolor='k', s=50)
ax13.set_xscale('log')
ax13.set_xlabel('$Z_{1.0}(m) - NZVM $', size=16)
ax13.set_ylabel('$V_{S30} (m/s)$', size=16)
ax13.legend(fontsize="16", frameon=False)
fig34,ax14  = plt.subplots(figsize=(19.2,9.83),constrained_layout=True)
scatter5  = ax14.scatter(df_site['Z1_Meas'],df_site['vs30'],label='',c='w')
mpl.cursor(scatter5,multiple=True).connect("add", hover_format)
for i, (name, group) in enumerate(grouped_dataproblematic):
    ax14.scatter(group['Z1_Meas'], group['vs30'],label=f"Cluster {name}", color=colors[i % len(colors)], edgecolor='k', s=50)
ax14.set_xscale('log')
ax14.set_xlabel('$Z_{1.0}(m)- database$', size=16)
ax14.set_ylabel('$V_{S30} (m/s)$', size=16)
ax14.legend(fontsize="16", frameon=False)
fig35,ax15  = plt.subplots(figsize=(19.2,9.83),constrained_layout=True)
scatter6  = ax15.scatter(df_site['T0'],df_site['vs30'],label='',c='w')
mpl.cursor(scatter6,multiple=True).connect("add", hover_format)
for i, (name, group) in enumerate(grouped_dataproblematic):
    ax15.scatter(group['T0'], group['vs30'],label=f"Cluster {name}", color=colors[i % len(colors)], edgecolor='k', s=50)
ax15.set_xscale('log')
ax15.set_xlabel('$T_0\  (s)$', size=16)
ax15.set_ylabel('$V_{S30} (m/s)$', size=16)
ax15.legend(fontsize="16", frameon=False)
#%% Histograms
# Vs30 for each cluster
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True,sharey=True)
logspace = False
if logspace:
    logbinsVs30 = np.logspace(np.log10(df_site['vs30'].min()),np.log10(df_site['vs30'].max()),10)
else:
    logbinsVs30 = np.linspace(min(df_site['vs30']), max(df_site['vs30']), 10)
axes = axes.flatten()
for i, (name, group) in enumerate(grouped_dataproblematic):
    ax = axes[i]
    ax.hist(group['vs30'], bins=logbinsVs30, weights = np.ones(len(group['vs30'])) / len(group['vs30']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size=16)
    mean = group['vs30'].mean()
    std_dev = group['vs30'].std()
    ax.tick_params(labelbottom=True,labelleft=True)
    if logspace:
        ax.set_xscale('log')
        ax.set_xticks([200,400,600,800,1000])    
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes[-num_cols:]:
    ax.set_xlabel('$V_{S30} (m/s)$', size=16)
fig.savefig(os.path.join('Histograms',f'{folderpath}','Vs30 Cluster Histogram.png'))
#%% T0 for each cluster
fig3, axes3 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
axes3 = axes3.flatten()
logspace = True
if logspace:
    logbinsT0 = np.logspace(np.log10(df_site['T0'].min()),np.log10(df_site['T0'].max()),7)
else:
    logbinsT0 = np.linspace(min(df_site['T0']), max(df_site['T0']), 10)
for i, (name, group) in enumerate(grouped_dataproblematic):
    ax = axes3[i]
    ax.hist(group['T0'],bins = logbinsT0, weights = np.ones(len(group['T0'])) / len(group['T0']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size=16)
    mean = group['T0'].mean()
    std_dev = group['T0'].std()
    ax.tick_params(labelbottom=True, labelleft = True)
    if logspace:
        ax.set_xscale('log')
        ax.set_xticks([0.05,0.1,0.5,1,2,3,6])    
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    ax.annotate(f'Mean: {mean:.1f}\nStd: {std_dev:.1f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes[-num_cols:]:
    ax.set_xlabel('$T_0 (s)$', size = 16)
fig3.savefig(os.path.join('Histograms',f'{folderpath}','T0 Cluster Histogram.png'))
#%% Geomorphology for each cluster
fig3, axes3 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
axes3 = axes3.flatten()
bins = np.array(['Basin','Basin-edge','Valley','Hill'])
for i, (name, group) in enumerate(grouped_dataproblematic):
    ax = axes3[i]
    counts = np.array([np.sum(group == bin_name) for bin_name in bins])
    percentage  = counts/np.sum(counts)
    percentage  = np.sum(percentage,axis=1)
    ax.bar(bins, percentage, color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    ax.tick_params(labelbottom=True, labelleft = True)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
for ax in axes3[-num_cols:]:
    ax.set_xlabel('Geomorphology', size = 16)
fig3.savefig(os.path.join('Histograms',f'{folderpath}','Geomorphology Cluster Histogram.png'))  
#%% Slope for each cluster
fig3, axes3 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
axes3 = axes3.flatten()
logspace = False
if logspace:
    logbinsslope = np.logspace(np.log10(df_site['Slope'].min()),np.log10(df_site['Slope'].max()),7)
else:
    logbinsslope = np.linspace(min(df_site['Slope']), max(df_site['Slope']), 11)
# logbinsslope = [0,0.01099,0.03099,0.05099,0.07099,0.10099,0.12099,0.15099,0.17099,0.20099,0.22099,0.25099,0.27099,0.30099,0.32]
for i, (name, group) in enumerate(grouped_dataproblematic):
    ax = axes3[i]
    ax.hist(group['Slope'], bins=logbinsslope, weights = np.ones(len(group['Slope'])) / len(group['Slope']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    mean = group['Slope'].mean()
    std_dev = group['Slope'].std()
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    ax.tick_params(labelbottom=True, labelleft = True)
    if logspace:
        ax.set_xscale('log')
        # ax.set_xticks([0.05,0.1,0.5,1,2,3,6])    
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    ax.annotate(f'Mean: {mean:.2f}\nStd: {std_dev:.2f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
    for ax in axes3[-num_cols:]:
        ax.set_xlabel('Slope (m/m)', size = 16)
fig3.savefig(os.path.join('Histograms',f'{folderpath}','Slope Cluster Histogram.png'))
#%% Z 1.0 for each cluster
fig4, axes4 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
logbinsZ1 =[0,10,20,30,50,100,200,500,1000,1500]
axes4 = axes4.flatten()
if len(grouped_dataproblematic) < n_clusters:
    axes4[len(grouped_dataproblematic)].remove()
for i, (name, group) in enumerate(grouped_dataproblematic):
    ax = axes4[i]
    ax.hist(group['Z1_NZVM'], bins=logbinsZ1, weights = np.ones(len(group['Z1_NZVM'])) / len(group['Z1_NZVM']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    mean = group['Z1_NZVM'].mean()
    std_dev = group['Z1_NZVM'].std()
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
    ax.set_xscale('log')
    ax.set_xticks([10,20,30,50,100,200,500,1000,1500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
for ax in axes4[-num_cols:]:
    ax.set_xlabel('Z1_NZVM', size = 16)   
fig4.savefig(os.path.join('Histograms',f'{folderpath}','Z1 NZVM Cluster Histogram.png'))
#%% Z 1.0 measured for each cluster
fig5, axes5 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
logbinsZ1 =[0,10,20,30,50,100,200,500,1000,1500]
axes5 = axes5.flatten()
if len(grouped_dataproblematic) < n_clusters:
    axes5[len(grouped_dataproblematic)].remove()
for i, (name, group) in enumerate(grouped_dataproblematic):
    ax = axes5[i]
    ax.hist(group['Z1_Meas'], bins=logbinsZ1, weights = np.ones(len(group['Z1_Meas'])) / len(group['Z1_Meas']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    mean = group['Z1_Meas'].mean()
    std_dev = group['Z1_Meas'].std()
    ax.set_xscale('log')
    ax.set_xticks([10,20,30,50,100,200,500,1000,1500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes5[-num_cols:]:
    ax.set_xlabel('Z1_database', size = 16)
fig5.savefig(os.path.join('Histograms',f'{folderpath}','Z1 database Cluster Histogram.png'))
#%% Ratio Z 1.0 for each cluster
fig6, axes6 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
axes6 = axes6.flatten()
if len(grouped_dataproblematic) < n_clusters:
    axes6[len(grouped_dataproblematic)].remove()
logbinsratio = [0,0.1,0.3,0.5,0.8,1,1.2,1.6,2,4,8,10,12]
for i, (name, group) in enumerate(grouped_dataproblematic):
    ax = axes6[i]
    ax.hist(group['Ratio'], bins=logbinsratio, weights = np.ones(len(group['Ratio'])) / len(group['Ratio']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    mean = group['Ratio'].mean()
    std_dev = group['Ratio'].std()
    ax.set_xscale('log')
    ax.set_xticks([0.1,0.3,0.5,0.8,1.2,2,4,8,12])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    # ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes6[-num_cols:]:
    ax.set_xlabel('Ratio (Z1_database/Z1_NZVM)', size = 16)
fig6.savefig(os.path.join('Histograms',f'{folderpath}','Z1 Ratio Cluster Histogram.png'))
#%% Roughness for each cluster (r370)
fig6, axes6 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
axes6 = axes6.flatten()
logspace = True
if logspace:
    logbinsr370 = np.logspace(np.log10(df_site['r370'].min()),np.log10(df_site['r370'].max()),10)
else:
    logbinsr370 = np.linspace(min(df_site['r370']), max(df_site['r370']), 11)
for i, (name, group) in enumerate(grouped_dataproblematic):
    ax = axes6[i]
    ax.hist(group['r370'], bins=logbinsr370, weights = np.ones(len(group['r370'])) / len(group['r370']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    mean = group['r370'].mean()
    std_dev = group['r370'].std()
    ax.set_xscale('log')
    ax.set_xticks([10,20,30,40,50,100,200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    # ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes6[-num_cols:]:
    ax.set_xlabel('Roughness (r370)', size = 16)
fig6.savefig(os.path.join('Histograms',f'{folderpath}','Roughness Cluster Histogram.png'))
fig7, axes7 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True,sharex=True, sharey=True)
axes7 = axes7.flatten()
logspace = True
if logspace:
    logbinsr1260 = np.logspace(np.log10(df_site['r1260'].min()),np.log10(df_site['r1260'].max()),10)
else:
    logbinsr1260 = np.linspace(min(df_site['r1260']), max(df_site['r1260']), 11)
for i, (name, group) in enumerate(grouped_dataproblematic):
    ax = axes7[i]
    ax.hist(group['r1260'], bins=logbinsr1260, weights = np.ones(len(group['r1260'])) / len(group['r1260']),color = colors[i % len(colors)], edgecolor='k')
    ax.set_title(f'Cluster {name}', size =16)
    if i % num_cols == 0:
        ax.set_ylabel('% sites', size = 16)
    mean = group['r1260'].mean()
    std_dev = group['r1260'].std()
    ax.set_xscale('log')
    ax.set_xticks([10,20,30,40,50,100,200,500,1000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(labelbottom=True, labelleft = True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    ax.annotate(f'N_sites = {len(group)}', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', fontweight='bold')
    # ax.annotate(f'Mean: {mean:.0f}\nStd: {std_dev:.0f}', xy=(0.88, 0.88), xycoords='axes fraction', fontsize=10, ha='right', va='top')
for ax in axes7[-num_cols:]:
    ax.set_xlabel('Roughness (r1260)', size = 16)
fig7.savefig(os.path.join('Histograms',f'{folderpath}','Roughness 1260 Cluster Histogram.png'))
#%% Plot the clusters
num_rows = (n_clusters + 2) // 3  
num_cols = 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True)
fig3, ax1 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig4, ax2 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig5, ax3 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig6, ax4 = plt.subplots(figsize=(12, 8),constrained_layout = True)

if num_rows == 1:
    axes = axes.reshape(1, -1)
elif num_cols == 1:
    axes = axes.reshape(-1, 1)

grouped_data = res_df.groupby(cluster_labels)
cluster_mean = res_df.groupby(cluster_labels).mean()
cluster_std  = res_df.groupby(cluster_labels).std()

for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
    # Plot the cluster's data in the current subplot
    ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color=colors[cluster_id-1],picker=True, pickradius=5, label=sorted(stat_names[cluster_id-1]))
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'k',linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)-cluster_data.values.T.std(axis=1),'k--',linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)+cluster_data.values.T.std(axis=1),'k--',linewidth=2)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    if (cluster_id-1) % num_cols == 0:
        ax.set_ylabel('${\it \delta S2S_s}$', size=12)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.set_title(f"Cluster {cluster_id}")
    ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    
# for ax in axes[2]:
#     ax.set_xlabel('Vibration Period, $\it T (s)$', size=12)
fig.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)
for i in range(n_clusters, num_rows*num_cols):
    fig.delaxes(axes.flatten()[i])

# fig.savefig(os.path.join('Optimal clusters','All clusters', f"Clusters = {n_clusters}, DTW = {DTW_radius}, smoothing = {use_smoothing}.png"))

n_clusterssoil = [3,4,6,8,9]
n_clustersrock = [1,2,5,7]
for cluster_id in n_clusterssoil:
    ax1.semilogx(cluster_mean.columns, cluster_mean.iloc[cluster_id-1, :], linewidth = 2,label = f"Cluster {cluster_id}", color=colors[cluster_id-1])
    ax1.set_xlim([0.01, 10])
    ax1.set_ylim([-1.0, 1.0])
    ax1.axhline(0,color='maroon',linestyle='--')
    ax1.set_ylabel('Mean in $\it \delta S2S_s$', size=12)
    ax1.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax1.tick_params(direction='in', axis='x', which='both')
    ax1.legend()
# fig3.savefig(os.path.join('Optimal clusters','Averages', f"Clusters = {n_clusters}_Soil.png")) 
for cluster_id in n_clustersrock:
    ax2.semilogx(cluster_mean.columns, cluster_mean.iloc[cluster_id-1, :], linewidth = 2,label = f"Cluster {cluster_id}", color=colors[cluster_id-1])
    ax2.set_xlim([0.01, 10])
    ax2.set_ylim([-1.0, 1.0])
    ax2.axhline(0,color='maroon',linestyle='--')
    ax2.set_ylabel('Mean in $\it \delta S2S_s$', size=12)
    ax2.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax2.tick_params(direction='in', axis='x', which='both')
    ax2.legend()
# fig4.savefig(os.path.join('Optimal clusters','Averages', f"Clusters = {n_clusters}_Rock.png"))     

ax3.semilogx(cluster_std.columns, phiS2S,linewidth =3,label = 'No clustering',color = 'k')
for cluster_id in n_clusterssoil:
    ax3.semilogx(cluster_std.columns, cluster_std.iloc[cluster_id-1, :], linewidth =2,label = f"Cluster {cluster_id}", color=colors[cluster_id-1])
    ax3.set_xlim([0.01, 10])
    ax3.set_ylabel('Std. deviation in $\it \delta S2S_s$', size=12)
    ax3.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax3.tick_params(direction='in', axis='x', which='both')
    ax3.legend()
# fig5.savefig(os.path.join('Optimal clusters','Standard devs', f"Clusters = {n_clusters}_Soil.png"))

ax4.semilogx(cluster_std.columns, phiS2S,linewidth =3,label = 'No clustering',color = 'k') 
for cluster_id in n_clustersrock:
    ax4.semilogx(cluster_std.columns, cluster_std.iloc[cluster_id-1, :], linewidth =2,label = f"Cluster {cluster_id}", color=colors[cluster_id-1])
    ax4.set_xlim([0.01, 10])
    ax4.set_ylabel('Std. deviation in $\it \delta S2S_s$', size=12)
    ax4.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax4.tick_params(direction='in', axis='x', which='both')
    ax4.legend()
# fig6.savefig(os.path.join('Optimal clusters','Standard devs', f"Clusters = {n_clusters}_Rock.png")) 
#%% Filtering out specific categories (Ex: Hill)
hillclusterlabels = np.array(df_site[df_site['Geom']=='Hill']['Cluster'])
hillstatids       = df_site[df_site['Geom']=='Hill']['stat_id'].apply(lambda x: f'Station_{x}')
hillres_df        = res_df.loc[hillstatids]
hillstatnames     = [None]*n_clusters  
for cluster_id in range(n_clusters):
    hillstats = hillres_df.index[hillclusterlabels==cluster_id+1]
    hillstats = [int(stations.rsplit("_",maxsplit=1)[-1]) for stations in hillstats]
    hillstatnames[cluster_id]    = [str(stations.loc[i-1].stat_name) for i in hillstats]
hillstations = pd.DataFrame({'stat_name':[item for sublist in hillstatnames for item in sublist],'Cluster':[index + 1 for index, sublist in enumerate(hillstatnames) for _ in sublist]})
hillculprit = pd.merge(hillstations,df_culpritsite[['stat_name','Cluster']],on='stat_name')
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True)
if num_rows == 1:
    axes = axes.reshape(1, -1)
elif num_cols == 1:
    axes = axes.reshape(-1, 1)

grouped_data = hillres_df.groupby(hillclusterlabels)
cluster_mean = hillres_df.groupby(hillclusterlabels).mean()
cluster_std  = hillres_df.groupby(hillclusterlabels).std()

for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
    # Plot the cluster's data in the current subplot
    ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color=colors[cluster_id-1],picker=True, pickradius=5,label= hillstatnames[cluster_id-1])
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'k',linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)-cluster_data.values.T.std(axis=1),'k--',linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)+cluster_data.values.T.std(axis=1),'k--',linewidth=2)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    if (cluster_id-1) % num_cols == 0:
        ax.set_ylabel('${\it \delta S2S_s}$', size=12)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.set_title(f"Cluster {cluster_id}")
    ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    
# for ax in axes[2]:
#     ax.set_xlabel('Vibration Period, $\it T (s)$', size=12)
fig.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)
for i in range(n_clusters, num_rows*num_cols):
    fig.delaxes(axes.flatten()[i])
#%% Hill clusters for Empirical residuals
hillclusterlabels = np.array(df_site[df_site['Geom']=='Hill']['Cluster'])
hillstatids       = df_site[df_site['Geom']=='Hill']['stat_id'].apply(lambda x: f'Station_{x}')
hillempres_df        = empres_df.loc[hillstatids]
hillstatnames     = [None]*n_clusters
for cluster_id in range(n_clusters):
    hillstats = hillempres_df.index[hillclusterlabels==cluster_id+1]
    hillstats = [int(stations.rsplit("_",maxsplit=1)[-1]) for stations in hillstats]
    hillstatnames[cluster_id]    = [str(stations.loc[i-1].stat_name) for i in hillstats]
hillstations = pd.DataFrame({'stat_name':[item for sublist in hillstatnames for item in sublist],'Cluster':[index + 1 for index, sublist in enumerate(hillstatnames) for _ in sublist]})
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True)
if num_rows == 1:
    axes = axes.reshape(1, -1)
elif num_cols == 1:
    axes = axes.reshape(-1, 1)

grouped_data = hillempres_df.groupby(hillclusterlabels)
cluster_mean = hillempres_df.groupby(hillclusterlabels).mean()
cluster_std  = hillempres_df.groupby(hillclusterlabels).std()

for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
    # Plot the cluster's data in the current subplot
    ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color=colors[cluster_id-1],picker=True, pickradius=5,label= hillstatnames[cluster_id-1])
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'k',linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)-cluster_data.values.T.std(axis=1),'k--',linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)+cluster_data.values.T.std(axis=1),'k--',linewidth=2)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    if (cluster_id-1) % num_cols == 0:
        ax.set_ylabel('${\it \delta S2S_s}$', size=12)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.set_title(f"Cluster {cluster_id}")
    ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    
# for ax in axes[2]:
#     ax.set_xlabel('Vibration Period, $\it T (s)$', size=12)
fig.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)
for i in range(n_clusters, num_rows*num_cols):
    fig.delaxes(axes.flatten()[i])
#%% Hill clusters for Fourier dS2S
hillclusterlabels = np.array(df_site[df_site['Geom']=='Hill']['Cluster'])
hillstatids       = df_site[df_site['Geom']=='Hill']['stat_id'].apply(lambda x: f'Station_{x}')
hillFdS2S_df        = FdS2S_df.loc[hillstatids]
hillstatnames     = [None]*n_clusters
for cluster_id in range(n_clusters):
    hillstats = hillFdS2S_df.index[hillclusterlabels==cluster_id+1]
    hillstats = [int(stations.rsplit("_",maxsplit=1)[-1]) for stations in hillstats]
    hillstatnames[cluster_id]    = [str(stations.loc[i-1].stat_name) for i in hillstats]
hillstations = pd.DataFrame({'stat_name':[item for sublist in hillstatnames for item in sublist],'Cluster':[index + 1 for index, sublist in enumerate(hillstatnames) for _ in sublist]})
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True)
if num_rows == 1:
    axes = axes.reshape(1, -1)
elif num_cols == 1:
    axes = axes.reshape(-1, 1)

grouped_data = hillFdS2S_df.groupby(hillclusterlabels)
cluster_mean = hillFdS2S_df.groupby(hillclusterlabels).mean()
cluster_std  = hillFdS2S_df.groupby(hillclusterlabels).std()

for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
    # Plot the cluster's data in the current subplot
    ax.semilogx(Ffreq, cluster_data.values.T,linewidth=1, color=colors[cluster_id-1],picker=True, pickradius=5,label= hillstatnames[cluster_id-1])
    ax.semilogx(Ffreq, cluster_data.values.T.mean(axis=1),'k',linewidth=2)
    ax.semilogx(Ffreq, cluster_data.values.T.mean(axis=1)-cluster_data.values.T.std(axis=1),'k--',linewidth=2)
    ax.semilogx(Ffreq, cluster_data.values.T.mean(axis=1)+cluster_data.values.T.std(axis=1),'k--',linewidth=2)
    ax.set_xlim([0.1, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    if (cluster_id-1) % num_cols == 0:
        ax.set_ylabel('${\it \delta S2S_s}$', size=12)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.set_title(f"Cluster {cluster_id}")
    ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    
for ax in axes[2]:
    ax.set_xlabel('Frequency, $\it f (Hz)$', size=12)
fig.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)
for i in range(n_clusters, num_rows*num_cols):
    fig.delaxes(axes.flatten()[i])
#%% Selected hill clusters
selected_clusters = [5, 1, 4, 6]
colorsnew = [colors[7],colors[2],'b','g']
hillclusterlabels = np.array(df_site[df_site['Geom']=='Hill']['Cluster'])
hillstatids       = df_site[df_site['Geom']=='Hill']['stat_id'].apply(lambda x: f'Station_{x}')
hillres_df        = res_df.loc[hillstatids]
grouped_data = hillres_df.groupby(hillclusterlabels)
num_selected_clusters = len(selected_clusters)
num_rows = (num_selected_clusters + 2) // 3
num_cols = 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18.22,9.86),constrained_layout=True,sharex=True,sharey=True)
if num_rows == 1:
    axes = axes.reshape(1, -1)
elif num_cols == 1:
    axes = axes.reshape(-1, 1)


# Loop through the selected clusters and their indices
for i, cluster_id in enumerate(selected_clusters):
    cluster_data = grouped_data.get_group(cluster_id)
    ax = axes.flatten()[i]

    # Plot the cluster's data in the current subplot
    ax.semilogx(cluster_data.columns, cluster_data.values.T, linewidth=1, color=colorsnew[i])
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1), 'k', linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1) - cluster_data.values.T.std(axis=1), 'k--', linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1) + cluster_data.values.T.std(axis=1), 'k--', linewidth=2)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0, color='maroon', linestyle='-')
    if i % num_cols == 0:
        ax.set_ylabel(r'$\mathregular{\delta S2S_s}$', size=50)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.tick_params(labelsize=24)
    ax.text(1.75,1.2,f"Cluster {i + 1}",fontsize=35, fontweight='bold')
    ax.text(1.2, -1.05, '$\mathregular{N_{sites}}$ = %d'%len(cluster_data),fontsize=40)
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes[0][0].text(0.013, 1.2, 'Underprediction', size=35)
axes[0][0].text(0.013, -1.39, 'Overprediction', size=35)
for ax in axes[1]:
    ax.set_xlabel('Vibration Period, T (s)', size=40)
#%% Comparison of simulation and empirical residuals based on clusters
num_rows = (n_clusters + 2) // 3  
num_cols = 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True)
fig3, ax1 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig4, ax2 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig5, ax3 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig6, ax4 = plt.subplots(figsize=(12, 8),constrained_layout = True)

if num_rows == 1:
    axes = axes.reshape(1, -1)
elif num_cols == 1:
    axes = axes.reshape(-1, 1)

grouped_data = res_df.groupby(cluster_labels)
emp_grouped_data = empres_df.groupby(cluster_labels)
cluster_mean = res_df.groupby(cluster_labels).mean()
cluster_std  = res_df.groupby(cluster_labels).std()
empcluster_mean = empres_df.groupby(cluster_labels).mean()
empcluster_std  = empres_df.groupby(cluster_labels).std()

for (cluster_id, cluster_data), ax in zip(emp_grouped_data, axes.flatten()):
    # Plot the cluster's data in the current subplot
    ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color=colors[cluster_id-1],picker=True, pickradius=5, label=sorted(stat_names[cluster_id-1]))
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'k',linewidth=2)
    ax.semilogx(cluster_mean.columns, cluster_mean.iloc[cluster_id-1, :],color = 'maroon',linewidth=2)
    # ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)-cluster_data.values.T.std(axis=1),'k--',linewidth=2)
    # ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)+cluster_data.values.T.std(axis=1),'k--',linewidth=2)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='#000075',linestyle='--')
    ax.set_ylabel('${\it \delta S2S_s}$', size=12)
    ax.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.set_title(f"Cluster {cluster_id}")
    ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    
fig.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)
for i in range(n_clusters, num_rows*num_cols):
    fig.delaxes(axes.flatten()[i])

# fig.savefig(os.path.join('SimVsEmp','Just emp.png'))
# fig.savefig(os.path.join('SimVsEmp','Emp & Sim Mean Comparison.png'))
n_clusterssoil = [3,4,6,8,9]
n_clustersrock = [1,2,5,7]
for cluster_id in n_clusterssoil:
    ax1.semilogx(cluster_mean.columns, cluster_mean.iloc[cluster_id-1, :], linewidth = 2,label = f"Cluster {cluster_id}_Simulation", color=colors[cluster_id-1])
    ax1.semilogx(empcluster_mean.columns, empcluster_mean.iloc[cluster_id-1, :], linewidth = 2, color=colors[cluster_id-1],linestyle = '--')
    ax1.set_xlim([0.01, 10])
    ax1.set_ylim([-1.0, 1.0])
    ax1.axhline(0,color='maroon',linestyle='--')
    ax1.set_ylabel('Mean in $\it \delta S2S_s$', size=12)
    ax1.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax1.tick_params(direction='in', axis='x', which='both')
    ax1.legend()
fig3.savefig(os.path.join('SimVsEmp','AveragesSoil.png'))   

for cluster_id in n_clustersrock:
    ax2.semilogx(cluster_mean.columns, cluster_mean.iloc[cluster_id-1, :], linewidth = 2,label = f"Cluster {cluster_id}_Simulation", color=colors[cluster_id-1])
    ax2.semilogx(empcluster_mean.columns, empcluster_mean.iloc[cluster_id-1, :], linewidth = 2, color=colors[cluster_id-1],linestyle = '--')
    ax2.set_xlim([0.01, 10])
    ax2.set_ylim([-1.0, 1.0])
    ax2.axhline(0,color='maroon',linestyle='--')
    ax2.set_ylabel('Mean in $\it \delta S2S_s$', size=12)
    ax2.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax2.tick_params(direction='in', axis='x', which='both')
    ax2.legend()
fig4.savefig(os.path.join('SimVsEmp','AveragesRock.png'))   

ax3.semilogx(cluster_std.columns, phiS2S,linewidth =3,label = 'No clustering_Sim',color = 'k')
ax3.semilogx(cluster_std.columns, phiS2S_emp,'k--',linewidth =3)

for cluster_id in n_clusterssoil:
    ax3.semilogx(cluster_std.columns, cluster_std.iloc[cluster_id-1, :], linewidth =2,label = f"Cluster {cluster_id}_Simulation", color=colors[cluster_id-1])
    ax3.semilogx(empcluster_std.columns, empcluster_std.iloc[cluster_id-1, :], linewidth =2, color=colors[cluster_id-1],linestyle = '--')
    ax3.set_xlim([0.01, 10])
    ax3.set_ylabel('Std. deviation in $\it \delta S2S_s$', size=12)
    ax3.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax3.tick_params(direction='in', axis='x', which='both')
    ax3.legend()
fig5.savefig(os.path.join('SimVsEmp','Stdsoil.png')) 

ax4.semilogx(cluster_std.columns, phiS2S,linewidth =3,label = 'No clustering_Sim',color = 'k')
ax4.semilogx(cluster_std.columns, phiS2S_emp,'k--',linewidth =3)
for cluster_id in n_clustersrock:
    ax4.semilogx(cluster_std.columns, cluster_std.iloc[cluster_id-1, :], linewidth =2,label = f"Cluster {cluster_id}_Simulation", color=colors[cluster_id-1])
    ax4.semilogx(empcluster_std.columns, empcluster_std.iloc[cluster_id-1, :], linewidth =2, color=colors[cluster_id-1],linestyle = '--')
    ax4.set_xlim([0.01, 10])
    ax4.set_ylabel('Std. deviation in $\it \delta S2S_s$', size=12)
    ax4.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax4.tick_params(direction='in', axis='x', which='both')
    ax4.legend()
fig6.savefig(os.path.join('SimVsEmp','StdRock.png')) 
#%% Clusters based on T0
num_rows = (n_clusters + 2) // 3  
num_cols = 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True)
if num_rows == 1:
    axes = axes.reshape(1, -1)
elif num_cols == 1:
    axes = axes.reshape(-1, 1)

grouped_data = res_df.groupby(cluster_labels)
color_values = df_site['T0']
norm = plt.Normalize(color_values.min(), color_values.max())
cmap = plt.get_cmap('jet')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
    ind = cluster_data.index
    ind = [int(statind.rsplit("_",maxsplit=1)[-1]) for statind in ind]
    T0s = [stations.loc[i-1,'T0'] for i in ind]
    colors = [cmap(norm(T0s[i])) for i in range(len(T0s))]
    for i, (index, data_row) in enumerate(cluster_data.iterrows()):
        ax.semilogx(data_row.index, data_row.values, linewidth=1, color=colors[i],picker=True, pickradius=5, label=stat_names[cluster_id-1][i])
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    ax.set_ylabel('${\it \delta S2S_s}$', size=12)
    ax.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.set_title(f"Cluster {cluster_id}")
    ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
cb = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.08)
cb.set_label('$ \it T_0\ (s)$', fontsize=20, fontweight = 'bold')
cb.ax.tick_params(labelsize=12)
fig.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)
#%% Clusters based on Vs30
fig3, axes3 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True)
color_values = df_site['vs30']
norm = plt.Normalize(color_values.min(), color_values.max())
cmap = plt.get_cmap('jet')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for (cluster_id, cluster_data), ax in zip(grouped_data, axes3.flatten()):
    ind = cluster_data.index
    ind = [int(statind.rsplit("_",maxsplit=1)[-1]) for statind in ind]
    T0s = [stations.loc[i-1,'vs30'] for i in ind]
    colors = [cmap(norm(T0s[i])) for i in range(len(T0s))]
    for i, (index, data_row) in enumerate(cluster_data.iterrows()):
        ax.semilogx(data_row.index, data_row.values, linewidth=1, color=colors[i],picker=True, pickradius=5, label=stat_names[cluster_id-1][i])
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    ax.set_ylabel('${\it \delta S2S_s}$', size=12)
    ax.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.set_title(f"Cluster {cluster_id}")
    ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
cb = fig3.colorbar(sm, ax=axes3.ravel().tolist(), pad=0.08)
cb.set_label('$ \it V_{S30}\ (m/s)$', fontsize=20, fontweight = 'bold')
cb.ax.tick_params(labelsize=12)
fig3.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)
#%% Clusters based on Z_1.0 NZVM
fig3, axes3 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True)
color_values = df_site['Z1_NZVM']
norm = plt.Normalize(color_values.min(), color_values.max())
cmap = plt.get_cmap('jet')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for (cluster_id, cluster_data), ax in zip(grouped_data, axes3.flatten()):
    ind = cluster_data.index
    ind = [int(statind.rsplit("_",maxsplit=1)[-1]) for statind in ind]
    T0s = [stations.loc[i-1,'Z1_NZVM'] for i in ind]
    colors = [cmap(norm(T0s[i])) for i in range(len(T0s))]
    for i, (index, data_row) in enumerate(cluster_data.iterrows()):
        ax.semilogx(data_row.index, data_row.values, linewidth=1, color=colors[i],picker=True, pickradius=5, label=stat_names[cluster_id-1][i])
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    ax.set_ylabel('${\it \delta S2S_s}$', size=12)
    ax.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.set_title(f"Cluster {cluster_id}")
    ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
cb = fig3.colorbar(sm, ax=axes3.ravel().tolist(), pad=0.08)
cb.set_label('$ \it Z_{1.0}\ (m)$', fontsize=20, fontweight = 'bold')
cb.ax.tick_params(labelsize=12)
fig3.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)
#%% Clusters based on geomorphology
fig3, axes3 = plt.subplots(num_rows, num_cols, figsize=(16.75, 9.66),constrained_layout = True)
category_values = stations['Geom']
cmap = ListedColormap(['red', 'blue', 'aqua', 'green'])
category_to_color = {'Basin': 0, 'Basin-edge': 1, 'Hill': 2, 'Valley': 3}
for (cluster_id, cluster_data), ax in zip(grouped_data, axes3.flatten()):
    for i, (index, data_row) in enumerate(cluster_data.iterrows()):
        ind = int(index.rsplit("_",maxsplit=1)[-1])
        ind = ind - 1
        category = category_values.loc[ind]
        color = cmap(category_to_color[category])
        ax.semilogx(data_row.index, data_row.values, linewidth=1, color=color,picker=True, pickradius=5, label=stat_names[cluster_id-1][i])
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    ax.set_ylabel('${\it \delta S2S_s}$', size=12)
    ax.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.set_title(f"Cluster {cluster_id}")
    ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Basin'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Basin-Edge'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', markersize=10, label='Hill'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Valley')]
# fig3.legend(handles=legend_elements, loc='center left')
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title='Categories')
fig3.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)
#%% Spatial plot
basemap = r"P:\mapplot\data\basemap\NZ10.tif"
fig3, ax3 = plt.subplots(figsize=(19.2,9.83),constrained_layout = True)
groups = df_site.groupby('Cluster')
for name, group in groups:
    ax3.scatter(group['lon'], group['lat'], marker = '^', color=colors[name-1], label=f"Cluster {name}")
sc = ax3.scatter(df_site['lon'],df_site['lat'], marker='^', color='w', alpha=0)
ax3.set_xlabel('Longitude',size=12)
ax3.set_ylabel('Latitude',size=12)
ax3.legend(frameon=False)
ax3.set_aspect('equal')
crs = 'EPSG:4326'
# ctx.add_basemap(ax3, crs=crs, source=basemap)
mpl.cursor(sc,multiple=True).connect("add", hover_format)
n_rows =  (n_clusters-1)//3+1
unique_clusters = df_site['Cluster'].unique()
fig4, ax4= plt.subplots(n_rows, 3, figsize = (19.2,9.83))
axes = ax4.flatten()
for i, cluster_id in enumerate(unique_clusters):
    clusterlatlong = df_site[df_site['Cluster'] == cluster_id]
    ax=axes[i]
    s2 = ax.scatter(clusterlatlong['lon'],clusterlatlong['lat'],marker = '^', color=colors[cluster_id-1])
    ax.set_xlabel('Longitude',size=12)
    ax.set_ylabel('Latitude',size=12) 
    ax.set_title(f'Cluster {cluster_id}')
    ax.text(173,-42,f"Count = {len(clusterlatlong)}")
    s2 = ax.scatter(df_site['lon'],df_site['lat'],marker = '^', color='w', alpha=0)
    mpl.cursor(s2,multiple=True).connect("add", hover_format)   
    # ctx.add_basemap(ax, crs=crs, source=basemap) 
for i in range(len(unique_clusters), n_rows * 3):
    fig4.delaxes(axes.flatten()[i])
fig4.tight_layout()

#%% Problematic sites
df_culpritcluster = pd.merge(df_culprits,df_site[['stat_name','Cluster','Geom']],on='stat_name')
df_culstats = pd.merge(df_culpritcluster,stations[['stat_name','stat_id']],on='stat_name')
df_culstats['newcolindex']=df_culstats['stat_id']-1
df_culstats['stat_id'] = 'Station_' + df_culstats['stat_id'].astype(str)
station_values = df_culstats['stat_id'].unique()
station_index  = df_culstats['newcolindex'].unique()
filteredres_df = res_df[res_df.index.isin(station_values)]
filteredempres_df = empres_df[empres_df.index.isin(station_values)]
#%% Drawing spatially problematic
fig5,ax5 = plt.subplots(figsize=(19.2,9.83),constrained_layout=True)
groups = df_culpritcluster.groupby('Cluster')
for name, group in groups:
    ax5.scatter(group['Longitude'], group['Latitude'], marker = '^', color=colors[name-1], label=f"Cluster {name}")
s3 = ax5.scatter(df_site['lon'],df_site['lat'], marker='^', color='w', alpha=0)
ax5.set_xlabel('Longitude',size=12)
ax5.set_ylabel('Latitude',size=12)
ax5.legend(frameon=False)
ax5.set_aspect('equal')
crs = 'EPSG:4326'
# ctx.add_basemap(ax5, crs=crs, source=basemap)
mpl.cursor(s3,multiple=True).connect("add", hover_format)
unique_clus = sorted(df_culpritcluster['Cluster'].unique())
fig6, ax6= plt.subplots(n_rows, 3, figsize = (19.2,9.83))
axes = ax6.flatten()
for i, cluster_id in enumerate(unique_clus):
    clusterculprit = df_culpritcluster[df_culpritcluster['Cluster'] == cluster_id]
    ax=axes[i]
    ax.scatter(clusterculprit['Longitude'],clusterculprit['Latitude'],marker = '^', color=colors[cluster_id-1])
    ax.set_xlabel('Longitude',size=12)
    ax.set_ylabel('Latitude',size=12) 
    ax.set_title(f'Cluster {cluster_id}')
    ax.text(173,-42,f"Count = {len(clusterculprit)}")
    s2 = ax.scatter(df_site['lon'],df_site['lat'],marker = '^', color='w', alpha=0)
    mpl.cursor(s2,multiple=True).connect("add", hover_format)   
    # ctx.add_basemap(ax, crs=crs, source=basemap) 
for i in range(len(unique_clusters), n_rows * 3):
    fig6.delaxes(axes.flatten()[i])
fig6.tight_layout()
#%% Plotting problematic sites clusters
fig7,ax7 = plt.subplots(num_rows, 3,figsize=(16.75, 9.66), constrained_layout=True)
fig8,ax8 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig9,ax9 = plt.subplots(figsize=(12, 8),constrained_layout = True)
culstats=[None]*n_clusters
axes=ax7.flatten()
for cluster_label in range(n_clusters):
    res_dfind = res_df.index[cluster_labels == cluster_label+1]
    commonind = filteredres_df.index.intersection(res_dfind)
    extracted = filteredres_df.loc[commonind]
    cul       = df_culpritcluster[df_culpritcluster['Cluster']==cluster_label+1]
    culstats[cluster_label]=cul['stat_name'].tolist()
    ax=axes[cluster_label]
    ax.semilogx(extracted.columns, extracted.values.T,linewidth=1,color=colors[cluster_label],picker=True, pickradius=5,label=sorted(culstats[cluster_label]))
    ax.semilogx(extracted.columns, extracted.values.T.mean(axis=1),'k',linewidth=2)
    ax.semilogx(extracted.columns, extracted.values.T.mean(axis=1)-extracted.values.T.std(axis=1),'k--',linewidth=2)
    ax.semilogx(extracted.columns, extracted.values.T.mean(axis=1)+extracted.values.T.std(axis=1),'k--',linewidth=2)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    ax.set_ylabel('${\it \delta S2S_s}$', size=12)
    ax.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.set_title(f"Cluster {cluster_label+1}")
    ax.text(1,1.3,f"Count = {len(extracted)}")
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    ax8.semilogx(extracted.columns, extracted.values.T.mean(axis=1),linewidth = 2,label = f"Cluster {cluster_label + 1}", color=colors[cluster_label])
    ax8.set_xlim([0.01, 10])
    ax8.set_ylim([-1.0, 1.0])
    ax8.axhline(0,color='maroon',linestyle='--')
    ax8.set_ylabel('Mean in $\it \delta S2S_s$', size=12)
    ax8.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax8.tick_params(direction='in', axis='x', which='both')
    ax8.legend()
    ax9.semilogx(extracted.columns, extracted.values.T.std(axis=1), linewidth =2,label = f"Cluster {cluster_label + 1}", color=colors[cluster_label])
    ax9.set_xlim([0.01, 10])
    ax9.set_ylabel('Std. deviation in $\it \delta S2S_s$', size=12)
    ax9.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax9.tick_params(direction='in', axis='x', which='both')
    ax9.legend()
    
fig7.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)
for i in range(n_clusters, num_rows*3):
    fig7.delaxes(axes.flatten()[i])
fig7.savefig(os.path.join('Optimal clusters','All clusters', f'{folderpath}',f"Culsites Clusters = {n_clusters}, DTW = {DTW_radius}, smoothing = {use_smoothing}.png"))
#%% Plotting problematic sites clusters (Division into rock and soil)
fig8,ax8 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig9,ax9 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig10,ax10 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig11,ax11 = plt.subplots(figsize=(12, 8),constrained_layout = True)
n_clusterssoil = [3,4,6,8,9]
n_clustersrock = [1,2,5,7]
culstats=[None]*n_clusters
for cluster_label in n_clusterssoil:
    res_dfind = res_df.index[cluster_labels == cluster_label]
    commonind = filteredres_df.index.intersection(res_dfind)
    extracted = filteredres_df.loc[commonind]
    cul       = df_culpritcluster[df_culpritcluster['Cluster']==cluster_label]
    culstats[cluster_label-1]=cul['stat_name'].tolist()
    ax8.semilogx(extracted.columns, extracted.values.T.mean(axis=1),linewidth = 2,label = f"Cluster {cluster_label}", color=colors[cluster_label-1])
    ax8.set_xlim([0.01, 10])
    ax8.set_ylim([-1.0, 1.0])
    ax8.axhline(0,color='maroon',linestyle='--')
    ax8.set_ylabel('Mean in $\it \delta S2S_s$', size=12)
    ax8.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax8.tick_params(direction='in', axis='x', which='both')
    ax8.legend()
    ax9.semilogx(extracted.columns, extracted.values.T.std(axis=1), linewidth =2,label = f"Cluster {cluster_label}", color=colors[cluster_label-1])
    ax9.set_xlim([0.01, 10])
    ax9.set_ylabel('Std. deviation in $\it \delta S2S_s$', size=12)
    ax9.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax9.tick_params(direction='in', axis='x', which='both')
    ax9.legend()
    
fig8.savefig(os.path.join('Optimal clusters','Averages', f'{folderpath}',f"Culsites Clusters = {n_clusters}_Soil.png"))
fig9.savefig(os.path.join('Optimal clusters','Standard devs', f'{folderpath}',f"Culsites Clusters = {n_clusters}_Soil.png"))

for cluster_label in n_clustersrock:
    res_dfind = res_df.index[cluster_labels == cluster_label]
    commonind = filteredres_df.index.intersection(res_dfind)
    extracted = filteredres_df.loc[commonind]
    cul       = df_culpritcluster[df_culpritcluster['Cluster']==cluster_label]
    culstats[cluster_label-1]=cul['stat_name'].tolist()
    ax10.semilogx(extracted.columns, extracted.values.T.mean(axis=1),linewidth = 2,label = f"Cluster {cluster_label}", color=colors[cluster_label-1])
    ax10.set_xlim([0.01, 10])
    ax10.set_ylim([-1.0, 1.0])
    ax10.axhline(0,color='maroon',linestyle='--')
    ax10.set_ylabel('Mean in $\it \delta S2S_s$', size=12)
    ax10.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax10.tick_params(direction='in', axis='x', which='both')
    ax10.legend()
    ax11.semilogx(extracted.columns, extracted.values.T.std(axis=1), linewidth =2,label = f"Cluster {cluster_label}", color=colors[cluster_label-1])
    ax11.set_xlim([0.01, 10])
    ax11.set_ylabel('Std. deviation in $\it \delta S2S_s$', size=12)
    ax11.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax11.tick_params(direction='in', axis='x', which='both')
    ax11.legend()

fig10.savefig(os.path.join('Optimal clusters','Averages', f'{folderpath}',f"Culsites Clusters = {n_clusters}_Rock.png"))
fig11.savefig(os.path.join('Optimal clusters','Standard devs', f'{folderpath}',f"Culsites Clusters = {n_clusters}_Rock.png"))
#%% Plotting problematic sites clusters with empirical residuals
fig7,ax7 = plt.subplots(num_rows, 3,figsize=(16.75, 9.66), constrained_layout=True)
culstats=[None]*n_clusters
axes=ax7.flatten()
for cluster_label in range(n_clusters):
    res_dfind = res_df.index[cluster_labels == cluster_label+1]
    commonind = filteredres_df.index.intersection(res_dfind)
    extracted = filteredres_df.loc[commonind]
    empres_dfind = empres_df.index[cluster_labels == cluster_label+1]
    commonindemp = filteredempres_df.index.intersection(empres_dfind)
    extractedemp = filteredempres_df.loc[commonindemp]
    cul       = df_culpritcluster[df_culpritcluster['Cluster']==cluster_label+1]
    culstats[cluster_label]=cul['stat_name'].tolist()
    ax=axes[cluster_label]
    ax.semilogx(extractedemp.columns, extractedemp.values.T,linewidth=1,color=colors[cluster_label],picker=True, pickradius=5,label=sorted(culstats[cluster_label]))
    ax.semilogx(extractedemp.columns, extractedemp.values.T.mean(axis=1),'k',linewidth=2)
    ax.semilogx(extracted.columns, extracted.values.T.mean(axis=1),color='maroon',linewidth=2)
    ax.semilogx(extractedemp.columns, extractedemp.values.T.mean(axis=1)-extractedemp.values.T.std(axis=1),'k--',linewidth=2)
    ax.semilogx(extractedemp.columns, extractedemp.values.T.mean(axis=1)+extractedemp.values.T.std(axis=1),'k--',linewidth=2)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    ax.set_ylabel('${\it \delta S2S_s}$', size=12)
    ax.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax.tick_params(direction='in', axis='x', which='both')
    ax.set_title(f"Cluster {cluster_label+1}")
    ax.text(1,1.3,f"Count = {len(extracted)}")
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    
fig7.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)
for i in range(n_clusters, num_rows*3):
    fig7.delaxes(axes.flatten()[i])
fig7.savefig(os.path.join('SimVsEmp', f'{folderpath}',f"Culsites Clusters = {n_clusters}.png"))
fig8,ax8 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig9,ax9 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig10,ax10 = plt.subplots(figsize=(12, 8),constrained_layout = True)
fig11,ax11 = plt.subplots(figsize=(12, 8),constrained_layout = True)
n_clusterssoil = [3,4,6,8,9]
n_clustersrock = [1,2,5,7]
culstats=[None]*n_clusters
for cluster_label in n_clusterssoil:
    res_dfind = res_df.index[cluster_labels == cluster_label]
    commonind = filteredres_df.index.intersection(res_dfind)
    extracted = filteredres_df.loc[commonind]
    empres_dfind = empres_df.index[cluster_labels == cluster_label]
    commonindemp = filteredempres_df.index.intersection(empres_dfind)
    extractedemp = filteredempres_df.loc[commonindemp]
    cul       = df_culpritcluster[df_culpritcluster['Cluster']==cluster_label]
    culstats[cluster_label-1]=cul['stat_name'].tolist()
    ax8.semilogx(extracted.columns, extracted.values.T.mean(axis=1),linewidth = 2,label = f"Cluster {cluster_label}_Simulation", color=colors[cluster_label-1])
    ax8.semilogx(extractedemp.columns, extractedemp.values.T.mean(axis=1),linewidth = 2,linestyle = '--', color=colors[cluster_label-1])
    ax8.set_xlim([0.01, 10])
    ax8.set_ylim([-1.0, 1.0])
    ax8.axhline(0,color='maroon',linestyle='--')
    ax8.set_ylabel('Mean in $\it \delta S2S_s$', size=12)
    ax8.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax8.tick_params(direction='in', axis='x', which='both')
    ax8.legend()
    ax9.semilogx(extracted.columns, extracted.values.T.std(axis=1), linewidth =2,label = f"Cluster {cluster_label}_Simulation", color=colors[cluster_label-1])
    ax9.semilogx(extractedemp.columns, extractedemp.values.T.std(axis=1),linewidth = 2,linestyle = '--', color=colors[cluster_label-1])
    ax9.set_xlim([0.01, 10])
    ax9.set_ylabel('Std. deviation in $\it \delta S2S_s$', size=12)
    ax9.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax9.tick_params(direction='in', axis='x', which='both')
    ax9.legend()
    
fig8.savefig(os.path.join('SimVsEmp', f'{folderpath}',"Averages_Soil.png"))
fig9.savefig(os.path.join('SimVsEmp', f'{folderpath}',"Std_Soil.png"))

for cluster_label in n_clustersrock:
    res_dfind = res_df.index[cluster_labels == cluster_label]
    commonind = filteredres_df.index.intersection(res_dfind)
    extracted = filteredres_df.loc[commonind]
    empres_dfind = empres_df.index[cluster_labels == cluster_label]
    commonindemp = filteredempres_df.index.intersection(empres_dfind)
    extractedemp = filteredempres_df.loc[commonindemp]
    cul       = df_culpritcluster[df_culpritcluster['Cluster']==cluster_label]
    culstats[cluster_label-1]=cul['stat_name'].tolist()
    ax10.semilogx(extracted.columns, extracted.values.T.mean(axis=1),linewidth = 2,label = f"Cluster {cluster_label}_Simulation", color=colors[cluster_label-1])
    ax10.semilogx(extractedemp.columns, extractedemp.values.T.mean(axis=1),linewidth = 2,linestyle = '--', color=colors[cluster_label-1])
    ax10.set_xlim([0.01, 10])
    ax10.set_ylim([-1.0, 1.0])
    ax10.axhline(0,color='maroon',linestyle='--')
    ax10.set_ylabel('Mean in $\it \delta S2S_s$', size=12)
    ax10.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax10.tick_params(direction='in', axis='x', which='both')
    ax10.legend()
    ax11.semilogx(extracted.columns, extracted.values.T.std(axis=1), linewidth =2,label = f"Cluster {cluster_label}_Simulation", color=colors[cluster_label-1])
    ax11.semilogx(extractedemp.columns, extractedemp.values.T.std(axis=1),linewidth = 2,linestyle = '--', color=colors[cluster_label-1])
    ax11.set_xlim([0.01, 10])
    ax11.set_ylabel('Std. deviation in $\it \delta S2S_s$', size=12)
    ax11.set_xlabel('Vibration Period, $\it T (s)$', size=12)
    ax11.tick_params(direction='in', axis='x', which='both')
    ax11.legend()

fig10.savefig(os.path.join('SimVsEmp', f'{folderpath}',"Averages_Rock.png"))
fig11.savefig(os.path.join('SimVsEmp', f'{folderpath}',"Std_Rock.png"))
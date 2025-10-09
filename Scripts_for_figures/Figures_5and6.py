# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:32:57 2024

@author: User
"""
from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
plt.close('all')

plt.rcParams["font.family"] = "Times New Roman"
import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_res_pSA_from_csv(im_data_ffp: Path):
    im_df = pd.read_csv(im_data_ffp, index_col=0).T
    pSA_columns = np.asarray([cur_col for cur_col in im_df.columns if cur_col.startswith("pSA")])
    im_df = im_df.loc[:, pSA_columns]
    pSA_period_values = [float(cur_c.rsplit("_", maxsplit=1)[-1]) for cur_c in pSA_columns]
    im_df.columns = pSA_period_values
    assert np.all(np.sort(pSA_period_values) == pSA_period_values)
    im_df = im_df.dropna()
    return im_df


flag = 1 # 1 for Simulation results and 0 for Empirical
varcomps       = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\PJSvarCompsBiased_sim.csv"))
phiS2S         = varcomps.iloc[6:,4]
sim_siteres    = load_res_pSA_from_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\New Results\Residuals\PJSreStationBiased_sim.csv"))
sim_sitereserr = load_res_pSA_from_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\New Results\Residuals\PJSreStationerror_sim.csv"))
emp_siteres    = load_res_pSA_from_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Empirical residuals\v22p11p10\Residuals\PJSreStationBiased_emp.csv"))
emp_sitereserr = load_res_pSA_from_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Empirical residuals\v22p11p10\Residuals\PJSreStationerror_emp.csv"))
Features       = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\9. Model development\Correlation\Features.csv"),index_col = 0)
Features       = Features.sort_values(by='stat_name')
newFeatures    = Features.copy()
Features.index = Features.index.map(lambda x: 'Station_' + str(x))
newFeatures.index = newFeatures.index.map(lambda x: 'stat_' + str(x))
cat_options    = ['Geomorphology','QVs30','Basin Type']
category       = 'QVs30'



if flag:
    grouped_data   = sim_siteres.groupby(Features[category])
    grouped_dataerr= sim_sitereserr.groupby(Features[category])
    cat_mean   = sim_siteres.groupby(Features[category]).mean()
    cat_std    = sim_siteres.groupby(Features[category]).std()
    cat_stdall = np.sqrt(np.mean((sim_siteres - np.mean(sim_siteres, axis=0))**2, axis=0) + np.mean(sim_sitereserr**2, axis=0))
    cat_phiS2S = {}
    cat_SE     = {}
    for label, group_errdS2S in grouped_dataerr:
        group_dS2S = grouped_data.get_group(label)
        # Cat_std =  np.sqrt(np.mean((group_dS2S-np.mean(group_dS2S, axis=0))**2,axis=0)+np.mean((group_errdS2S-np.mean(group_dS2S, axis=0))**2,axis=0))
        Cat_std =  np.sqrt(np.mean((group_dS2S-np.mean(group_dS2S, axis=0))**2,axis=0)+np.mean((group_errdS2S)**2,axis=0))
        # Cat_std =  np.sqrt(np.mean((group_dS2S)**2,axis=0)+np.mean((group_errdS2S)**2,axis=0))
        Cat_SE  = Cat_std/np.sqrt(len(group_dS2S))
        cat_phiS2S[label]= Cat_std
        cat_SE[label] = Cat_SE
    cat_phiS2S  = pd.DataFrame(cat_phiS2S)
    cat_phiS2S  = cat_phiS2S.transpose()
    cat_SE  = pd.DataFrame(cat_SE)
    cat_SE  = cat_SE.transpose()
    Nj = Features[category].value_counts()
    Nj_minus_1 = Nj - 1
    global_std_devs = {}
    for column in cat_phiS2S.columns:
        subset_variances = cat_phiS2S[column] ** 2
        numerator = ((Nj_minus_1 * subset_variances).sum())
        denominator = Nj_minus_1.sum()
        global_variance = numerator / denominator
        global_std_devs[column] = global_variance ** 0.5 
    global_std_devs = pd.Series(global_std_devs)
    out_dir    = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Journal papers\Paper 1\Figures\Figure Outputs\Simulation")
else:
    grouped_data   = emp_siteres.groupby(Features[category])
    grouped_dataerr= emp_sitereserr.groupby(Features[category])
    cat_mean   = emp_siteres.groupby(Features[category]).mean()
    cat_std    = emp_siteres.groupby(Features[category]).std()
    cat_stdall = np.sqrt(np.mean(emp_siteres**2,axis=0)+np.mean(emp_sitereserr**2,axis=0))
    cat_phiS2S = {}
    for label, group_errdS2S in grouped_dataerr:
        group_dS2S = grouped_data.get_group(label)
        Cat_std =  np.sqrt(np.mean(group_dS2S**2,axis=0)+np.mean(group_errdS2S**2,axis=0))
        cat_phiS2S[label]= Cat_std
    cat_phiS2S  = pd.DataFrame(cat_phiS2S)
    cat_phiS2S  = cat_phiS2S.transpose()
    out_dir    = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Journal papers\Paper 1\Figures\Figure Outputs\Empirical")

if category == "Geomorphology":
    desired_order  = ['Basin','Basin Edge','Valley','Hill']
    grouped_data = sorted(grouped_data, key=lambda x: desired_order.index(x[0]))
    # colors = ['b','#006400','#4b0082','#800000']
    colors = ['#1E90FF', '#66C266', '#9370DB', '#993333']
elif category == "Basin Type":
    desired_order = ["Non-Basin","Unmodeled Basin","Type 1 Basin","Type 3 Basin","Type 4 Basin"]
    grouped_data = sorted(grouped_data, key=lambda x: desired_order.index(x[0]))
    colors = ['#800000','#e6194B','#000075','#3cb44b','#808000']
else:
    colors = ['#ff7f0e', '#17becf', '#9467bd'] 
#%% All graphs - Individual curves, mean and standard deviation
num_cat        = len(Features[category].unique())
n_rows         = (num_cat + 1) // 2
n_cols         = 2
i              = 1

fig,axes         = plt.subplots(n_rows,n_cols, figsize = (10.09,  8.1),constrained_layout=True,sharex=True,sharey=True)
fig.text(0.0, 1, '(a)', fontsize=26, fontweight='bold', ha='left', va='top')
for (cat_id, cat_data), ax in zip(grouped_data, axes.flatten()):
    # Plot the cat's data in the current subplot
    ax.semilogx(cat_data.columns, cat_data.values.T,linewidth=1, color='#a9a9a9',alpha = 0.4)
    ax.semilogx(cat_data.columns, cat_mean.loc[cat_id],color = colors[i-1],linewidth=2)
    ax.semilogx(cat_data.columns, cat_mean.loc[cat_id]-cat_phiS2S.loc[cat_id],color=colors[i-1],linestyle='--',linewidth=2)
    ax.semilogx(cat_data.columns, cat_mean.loc[cat_id]+cat_phiS2S.loc[cat_id],color=colors[i-1],linestyle='--',linewidth=2)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='k',linestyle='--')
    if (i-1) % n_cols == 0:
        ax.set_ylabel('${\it \delta S2S_s}$', size=26)
    ax.tick_params(labelsize=20,direction='in', axis='both', which='both')
    ax.text(0.0105,1.25,f"{cat_id} (N = {len(cat_data)})",fontweight='bold',size=22)
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha = 0.5)
    i = i+1
axes[0][0].text(0.2,0.98,'Underprediction',size=24,fontstyle='italic')
axes[0][0].text(0.25,-1.2,'Overprediction',size=24,fontstyle='italic')
  
for ax in axes[n_rows-1]:
    ax.set_xlabel('Vibration Period, T (s)', size=26) 

for i in range(num_cat, n_rows*n_cols):
    fig.delaxes(axes.flatten()[i])
# Basin Edge subplot (top-right)
ax_edge = axes[0][1]

# Arrow to solid green line (category mean)
ax_edge.annotate(
    "Category mean",
    xy=(0.38, 0.22),       # point on the solid line
    xytext=(0.1, 0.85),      # text position
    fontsize=24,
    arrowprops=dict(arrowstyle="->", lw=1, color="black"),
    ha="left", va="center"
)

# Arrow to dashed green line (68% CI)
ax_edge.annotate(
     "Category mean ± Category std",
    xy=(0.30, -0.04),       # point on upper dashed line
    xytext=(0.32, -0.75),      # text position
    fontsize=24,
    arrowprops=dict(arrowstyle="->", lw=1, color="black"),
    ha="center", va="center"
)
axes[1][0].tick_params(axis='x', pad=5)
axes[1][1].tick_params(axis='x', pad=5)
fig.savefig(os.path.join(out_dir,f"All {category} ind.pdf"))
#%% Mean and standard deviation in two subplots of a plot - Geomorphology
fig,axes    = plt.subplots(2,1,figsize = (10.09,  8.1),constrained_layout=True,sharex=True)
i = 0
fig.text(0.0, 1, '(b)', fontsize=26, fontweight='bold', ha='left', va='top')
axes[1].semilogx(cat_data.columns, cat_stdall,color='k',label='All sites',linewidth=2)

for (cat_id, cat_data) in grouped_data:
    axes[0].semilogx(cat_data.columns, cat_mean.loc[cat_id],color=colors[i])
    axes[0].fill_between(cat_data.columns, cat_mean.loc[cat_id]-cat_SE.loc[cat_id], cat_mean.loc[cat_id]+cat_SE.loc[cat_id],color=colors[i],alpha=0.1)
    # axes[1].semilogx(cat_data.columns, cat_phiS2S.loc[cat_id],color=colors[i],label=f'{cat_id} sites')
    axes[1].semilogx(cat_data.columns, cat_phiS2S.loc[cat_id],color=colors[i],label=f'{cat_id} (N = {len(cat_data)})')
    i = i + 1
    

axes[0].axhline(0,color='k',linestyle='--',linewidth=2)
axes[0].set_ylabel(r'$\overline{\delta S2S_{\mathrm{cat}}}$', size=26)
axes[0].tick_params(labelsize=20,direction='in', axis='both', which='both')
axes[0].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes[0].set_ylim([-0.45,0.45]) 
axes[0].set_xlim([0.01, 10]) 
axes[0].annotate(
     "Category standard error",
    xy=(0.64, 0.24),       
    xytext=(0.65, 0.35),     
    fontsize=24,
    arrowprops=dict(arrowstyle="->", lw=1, color="black"),
    ha="center", va="center"
)
axes[1].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
axes[1].set_ylabel('$\phi_{S2S,cat}$',size=26)
axes[1].set_xlabel('Vibration Period, T (s)', size=26) 
axes[1].legend(fontsize=24,loc='lower left', bbox_to_anchor=(0, -0.038),ncol=2,columnspacing=0.01,handletextpad=0.3)
axes[1].tick_params(labelsize=20,direction='in', axis='both', which='both')
axes[1].set_xlim([0.01, 10])
axes[1].set_ylim([0,0.62])  
axes[1].tick_params(axis='x', pad=5)
# fig.savefig(os.path.join(out_dir,f"All {category} Mean & Std.pdf"))
#%% Mean and standard deviation in two subplots of a plot - QVs30 and Basin Type
fig,axes    = plt.subplots(2,1,figsize = (10.09,  8.1),constrained_layout=True,sharex=True)
i = 0
fig.text(0.0, 1, '(b)', fontsize=26, fontweight='bold', ha='left', va='top')
axes[1].semilogx(cat_data.columns, cat_stdall,color='k',label='All sites',linewidth=2)

for (cat_id, cat_data) in grouped_data:
    axes[0].semilogx(cat_data.columns, cat_mean.loc[cat_id],color=colors[i])
    axes[0].fill_between(cat_data.columns, cat_mean.loc[cat_id]-cat_SE.loc[cat_id], cat_mean.loc[cat_id]+cat_SE.loc[cat_id],color=colors[i],alpha=0.1)
    # axes[1].semilogx(cat_data.columns, cat_phiS2S.loc[cat_id],color=colors[i],label=f'{cat_id} sites')
    axes[1].semilogx(cat_data.columns, cat_phiS2S.loc[cat_id],color=colors[i],label=f'{cat_id} (N = {len(cat_data)})')
    i = i + 1
    

axes[0].axhline(0,color='k',linestyle='--',linewidth=2)
axes[0].text(0.0105,0.35,'Underprediction',size=26,fontstyle='italic')
axes[0].text(0.0105,-0.39,'Overprediction',size=26,fontstyle='italic')
axes[0].set_ylabel(r'$\overline{\delta S2S_{\mathrm{cat}}}$', size=26)
axes[0].tick_params(labelsize=20,direction='in', axis='both', which='both')
axes[0].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
axes[0].set_ylim([-0.45,0.45]) 
axes[0].set_xlim([0.01, 10]) 
axes[0].annotate(
     "Category standard error",
    xy=(2, 0.15),       # point on upper dashed line
    xytext=(1.5, 0.25),      # text position
    fontsize=24,
    arrowprops=dict(arrowstyle="->", lw=1, color="black"),
    ha="center", va="center"
)
axes[0].annotate(
     "Category mean",
    xy=(3.58,-0.04),       # point on upper dashed line
    xytext=(2.09, -0.25),      # text position
    fontsize=24,
    arrowprops=dict(arrowstyle="->", lw=1, color="black"),
    ha="center", va="center"
)
axes[1].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
axes[1].set_ylabel('$\phi_{S2S,cat}$',size=26)
axes[1].set_xlabel('Vibration Period, T (s)', size=26) 
axes[1].legend(fontsize=22,loc='lower left', bbox_to_anchor=(0, -0.035))
# axes[1].legend(fontsize=22,loc='lower left', bbox_to_anchor=(0, -0.035),ncol=2,columnspacing=0.1,handletextpad=0.3)
# axes[1].legend(fontsize=17,loc='lower left') # For geomorphology
axes[1].tick_params(labelsize=20,direction='in', axis='both', which='both')
axes[1].set_xlim([0.01, 10])
axes[1].set_ylim([0,0.62])    
axes[1].tick_params(axis='x', pad=5)
fig.savefig(os.path.join(out_dir,f"All {category} Mean & Std.pdf"))
#%%
# --- Filter only Basin Edge sites excluding Unmodeled Basin ---
basin_edge_mask = (Features["Geomorphology"] == "Basin Edge") & (Features["Basin Type"] != "Unmodeled Basin")
basin_edge_sites = Features[basin_edge_mask].index

# Subset residuals and errors
be_siteres = sim_siteres.loc[basin_edge_sites]
be_sitereserr = sim_sitereserr.loc[basin_edge_sites]

# Compute mean and std
be_mean = be_siteres.mean()
be_std = np.sqrt(np.mean((be_siteres - np.mean(be_siteres, axis=0))**2, axis=0) +
                 np.mean(be_sitereserr**2, axis=0))
be_SE = be_std / np.sqrt(len(be_siteres))

boot_means = []
for _ in range(1000):
    sample = be_siteres.sample(n=len(be_siteres), replace=True)
    boot_means.append(sample.mean())
    
boot_means = pd.DataFrame(boot_means)
be_mean_boot = boot_means.mean()
be_std_boot = boot_means.std()
# --- Make separate plot ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True, sharex=True)

axes[0].semilogx(be_siteres.columns, be_mean, color="#66C266")
axes[0].semilogx(be_siteres.columns, be_mean_boot, color="r")
axes[0].fill_between(be_siteres.columns, be_mean - 2*be_SE, be_mean + 2*be_SE,
                     color="#66C266", alpha=0.2)
axes[0].axhline(0, color="k", linestyle="--", linewidth=1)
axes[0].set_ylabel(r'$\overline{\delta S2S_{\mathrm{Basin\ Edge}}}$', size=22)
axes[0].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes[0].tick_params(labelsize=18, direction='in')

axes[1].semilogx(be_siteres.columns, be_std, color="#66C266",
                 label=f"Basin Edge (N = {len(be_siteres)})")
axes[1].semilogx(be_siteres.columns, be_std_boot, color="r",
                 label=f"Basin Edge (N = {len(be_siteres)})")
axes[1].semilogx(be_siteres.columns, be_SE, color="b",
                 label=f"Basin Edge (N = {len(be_siteres)})")
axes[1].set_xlabel("Vibration Period, T (s)", size=22)
axes[1].set_ylabel(r'$\phi_{S2S,\mathrm{Basin\ Edge}}$', size=22)
axes[1].set_xlim([0.01, 10])
axes[1].set_ylim([0, 0.62])
axes[1].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes[1].tick_params(labelsize=18, direction='in')
axes[1].legend(fontsize=14, loc="lower left")

fig.savefig(os.path.join(out_dir, "BasinEdge_noUnmodeledBasin.pdf"))
#%%
fig, axes = plt.subplots(figsize=(10, 8), constrained_layout=True, sharex=True)
summary_stats = []
axes.semilogx(be_siteres.columns, be_mean_boot, color="#66C266")
axes.fill_between(be_siteres.columns, be_mean_boot - 2*be_std_boot, be_mean + 2*be_std_boot,
                     color="#66C266", alpha=0.2)
axes.axhline(0, color="k", linestyle="--", linewidth=1)
axes.set_ylabel(r'$\overline{\delta S2S_{\mathrm{Basin\ Edge}}}$', size=22)
axes.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes.tick_params(labelsize=18, direction='in')
axes.set_xlabel("Vibration Period, T (s)", size=22)
axes.set_ylabel(r'$\phi_{S2S,\mathrm{Basin\ Edge}}$', size=22)
axes.set_xlim([0.01, 10])
axes.set_ylim([-0.6, 0.6])
axes.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes.tick_params(labelsize=18, direction='in')
summary_df = pd.DataFrame({
    'IM_sel': be_siteres.columns,
    'mean': be_mean_boot,
    'sigma_mean': be_std_boot
}).set_index('IM_sel')
save_dir = r'C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\9. Model development\Model Diagnostics\ModelsHillSites\M10FinalSlopeModelv2'
summary_df.to_excel(os.path.join(save_dir,"Bootstrapping_results_BE.xlsx"))
fig.savefig(os.path.join(out_dir, "BasinEdge_noUnmodeledBasin2std.pdf"))



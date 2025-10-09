# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:51:26 2024

@author: User
"""

import numpy as np
import pandas as pd
import addcopyfighandler
from pathlib import Path
import matplotlib.pyplot as plt
import os
data_dir1           = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\New Results")
ObsIMs              = pd.read_csv(Path(data_dir1 / "im_obs.csv"))
SimIMs              = pd.read_csv(data_dir1 / "im_sim.csv")
dS2Sres_all         = pd.read_csv(data_dir1/ "Residuals"/ "PJSreStationBiased_sim.csv",index_col=0)
dS2Serr_all         = pd.read_csv(data_dir1/"Residuals"/ "PJSreStationerror_sim.csv",index_col=0)
varcomps_all        = pd.read_csv(data_dir1/"Residuals"/ "PJSvarCompsBiased_sim.csv",index_col=0)
Features            = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\9. Model development\Correlation\FeaturesmodPaper1.csv"),index_col = 'Station Code')
Features.index      = 'Station_' + Features.index.astype(str)
stations            = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\stations.csv"))
stations            = pd.merge(stations[['stat_id','stat_name']],Features[['stat_name','Basin Type','Vs30','Z1','Distance to basin edge','Slope','Geomorphology','T0','T0new','T0final']],on='stat_name')
phiS2S_RS           = varcomps_all.iloc[6:,3]
T_RS                = varcomps_all.iloc[6:,0]
dS2Sres_all         = dS2Sres_all.dropna(axis=1)
dS2Sres_all         = dS2Sres_all.transpose()
dS2Serr_all         = dS2Serr_all.dropna(axis=1)
dS2Serr_all         = dS2Serr_all.transpose()
ostationsmod        = stations.copy()
ostationsmod['stat_id'] = 'Station_'+ ostationsmod['stat_id'].astype(str)
dS2Sres_all         = dS2Sres_all.iloc[:,6:]
dS2Serr_all         = dS2Serr_all.iloc[:,6:]
dS2Ssitedf          = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\9. Model development\Correlation\PJSreStationBiased_sim_TableauTest.csv"))
# out_dir             = Path(os.path.join(os.getcwd(),"Model Diagnostics","ModelsBasinSites","Anydependency"))
out_dir = Path(r'C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Journal papers\Paper 1\Figures\Figure Outputs\Residual analysis')
#%% LOWESS function define
def loess(x, y, f): # adapted from https://james-brennan.github.io/posts/lowess_conf/
    """
    Basic LOWESS smoother with uncertainty. 
    Note:
        - Not robust (so no iteration) and
             only normally distributed errors. 
        - No higher order polynomials d=1 
            so linear smoother.
    """
    # get some paras
    xwidth = f*(x.max()-x.min()) # effective width after reduction factor
    N = len(x) # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)
    # define the weigthing function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)
    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest 
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 * 
                                A[i].dot(np.linalg.inv(ATA)
                                                    ).dot(A[i]))
    return y_sm, y_stderr
#%% Residual analysis - Paper 1 - Basin sites ONLY - T0
sitedfBasin      = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Basin") & (dS2Ssitedf['Basin Type']!="Unmodeled")]
T0Basin          = stations[(stations['Geomorphology']=="Basin") & (stations['Basin Type']!="Unmodeled")]["T0"]
dS2SbasinUn      = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Basin") & (ostationsmod['Basin Type'] != "Unmodeled")]['stat_id']]
colors           = ['#000075','#3cb44b','#808000']
unique_type      = sitedfBasin['Basin Type'].unique()
custom_order     = ['Type 1','Type 2','Type 3','Type 4']
mask         = T0Basin.notna()
x            = T0Basin[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_type  = sorted(unique_type, key=lambda x: custom_order.index(x))
IMs              = ['pSA_0.869749002618','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_type):
        ax[i,j].scatter(sitedfBasin[sitedfBasin['Basin Type']==geom]['T0'],sitedfBasin[sitedfBasin['Basin Type']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xlim([0.1,10])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('$T_0$ (s)', size=20)
ax[1][1].set_xlabel('$T_0$ (s)', size=20)
ax[1][0].tick_params(axis='x', pad=10)
ax[1][1].tick_params(axis='x', pad=10)
plt.savefig(os.path.join(out_dir,"BasinsResAnalysis.pdf"))
#%% Residual analysis - Paper 1 - Basin sites ONLY - Distance to Basin-edge
sitedfBasin      = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Basin")& (dS2Ssitedf['Basin Type']!="Unmodeled")]
T0Basin          = stations[(stations['Geomorphology']=="Basin")& (stations['Basin Type']!="Unmodeled")]["Distance to basin edge"]
dS2SBasinUn      = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Basin")& (ostationsmod['Basin Type'] != "Unmodeled")]['stat_id']]
colors           = ['#000075','#3cb44b','#808000']
unique_type      = sitedfBasin['Basin Type'].unique()
custom_order     = ['Type 1','Type 2','Type 3','Type 4']
mask         = T0Basin.notna()
x            = T0Basin[mask]
x            = np.array(x)
y            = dS2SBasinUn
order        = np.argsort(x)
unique_type  = sorted(unique_type, key=lambda x: custom_order.index(x))
IMs              = ['pSA_0.869749002618','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.5)
    for k, geom in enumerate(unique_type):
        ax[i,j].scatter(sitedfBasin[sitedfBasin['Basin Type']==geom]['Distance to basin edge'],sitedfBasin[sitedfBasin['Basin Type']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    # ax[i, j].set_xlim([10,300])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('Distance to basin edge (m)', size=20)
ax[1][1].set_xlabel('Distance to basin edge (m)', size=20)
ax[1][0].tick_params(axis='x', pad=10)
ax[1][1].tick_params(axis='x', pad=10)
plt.savefig(os.path.join(out_dir,"BasinResAnalysisDBE.pdf"))
#%% Residual analysis - Paper 1 - Basin sites ONLY - Vs30
sitedfBasin      = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Basin") & (dS2Ssitedf['Basin Type']!="Unmodeled")]
T0Basin          = stations[(stations['Geomorphology']=="Basin") & (stations['Basin Type']!="Unmodeled")]["Vs30"]
dS2SbasinUn      = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Basin") & (ostationsmod['Basin Type'] != "Unmodeled")]['stat_id']]
colors           = ['#000075','#3cb44b','#808000']
unique_type      = sitedfBasin['Basin Type'].unique()
custom_order     = ['Type 1','Type 2','Type 3','Type 4']
mask         = T0Basin.notna()
x            = T0Basin[mask]
x            = np.array(x)
y            = dS2SBasinUn
order        = np.argsort(x)
unique_type  = sorted(unique_type, key=lambda x: custom_order.index(x))
IMs              = ['pSA_0.869749002618','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_type):
        ax[i,j].scatter(sitedfBasin[sitedfBasin['Basin Type']==geom]['Vs30'],sitedfBasin[sitedfBasin['Basin Type']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    # ax[i, j].set_xlim([10,300])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('$V_{S30}$ (m/s)', size=20)
ax[1][1].set_xlabel('$V_{S30}$ (m/s)', size=20)
ax[1][0].tick_params(axis='x', pad=10)
ax[1][1].tick_params(axis='x', pad=10)
plt.savefig(os.path.join(out_dir,"BasinResAnalysisVs30.pdf"))
#%% Residual analysis - Paper 1 - Basin sites ONLY - Z1
sitedfBasin      = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Basin")& (dS2Ssitedf['Basin Type']!="Unmodeled")]
T0Basin          = stations[(stations['Geomorphology']=="Basin")& (stations['Basin Type']!="Unmodeled")]["Z1"]
dS2SBasinUn      = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Basin")& (ostationsmod['Basin Type'] != "Unmodeled")]['stat_id']]
colors           = ['#000075','#3cb44b','#808000']
unique_type      = sitedfBasin['Basin Type'].unique()
custom_order     = ['Type 1','Type 2','Type 3','Type 4']
mask         = T0Basin.notna()
x            = T0Basin[mask]
x            = np.array(x)
y            = dS2SBasinUn
order        = np.argsort(x)
unique_type  = sorted(unique_type, key=lambda x: custom_order.index(x))
IMs              = ['pSA_0.869749002618','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_type):
        ax[i,j].scatter(sitedfBasin[sitedfBasin['Basin Type']==geom]['Z1'],sitedfBasin[sitedfBasin['Basin Type']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    # ax[i, j].set_xlim([10,300])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('$Z_{1.0}$ (m)', size=20)
ax[1][1].set_xlabel('$Z_{1.0}$ (m)', size=20)
ax[1][0].tick_params(axis='x', pad=10)
ax[1][1].tick_params(axis='x', pad=10)
plt.savefig(os.path.join(out_dir,"BasinResAnalysisZ1.pdf"))
#%% Residual analysis - Paper 1 - Valley sites ONLY - T0
sitedfValley      = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley")& (dS2Ssitedf['Basin Type']!="Unmodeled")]
T0Valley          = stations[(stations['Geomorphology']=="Valley")& (stations['Basin Type']!="Unmodeled")]["T0"]
dS2SValleyUn      = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley")& (ostationsmod['Basin Type'] != "Unmodeled")]['stat_id']]
colors           = ['#000075','#3cb44b','#808000']
unique_type      = sitedfValley['Basin Type'].unique()
custom_order     = ['Type 1','Type 2','Type 3','Type 4']
mask         = T0Valley.notna()
x            = T0Valley[mask]
x            = np.array(x)
y            = dS2SValleyUn
order        = np.argsort(x)
unique_type  = sorted(unique_type, key=lambda x: custom_order.index(x))
IMs              = ['pSA_0.869749002618','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_type):
        ax[i,j].scatter(sitedfValley[sitedfValley['Basin Type']==geom]['T0'],sitedfValley[sitedfValley['Basin Type']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xlim([0.1,3])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('$T_0$ (s)', size=20)
ax[1][1].set_xlabel('$T_0$ (s)', size=20)
ax[1][0].tick_params(axis='x', pad=10)
ax[1][1].tick_params(axis='x', pad=10)
plt.savefig(os.path.join(out_dir,"ValleysResAnalysis.pdf"))
#%% Residual analysis - Paper 1 - Valley sites ONLY - Distance to Basin-edge
sitedfValley      = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley")& (dS2Ssitedf['Basin Type']!="Unmodeled")]
T0Valley          = stations[(stations['Geomorphology']=="Valley")& (stations['Basin Type']!="Unmodeled")]["Distance to basin edge"]
dS2SValleyUn      = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley")& (ostationsmod['Basin Type'] != "Unmodeled")]['stat_id']]
colors           = ['#e6194B','#000075','#3cb44b','#808000']
unique_type      = sitedfValley['Basin Type'].unique()
custom_order     = ['Unmodeled','Type 1','Type 2','Type 3','Type 4']
mask         = T0Valley.notna()
x            = T0Valley[mask]
x            = np.array(x)
y            = dS2SValleyUn
order        = np.argsort(x)
unique_type  = sorted(unique_type, key=lambda x: custom_order.index(x))
IMs              = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_type):
        ax[i,j].scatter(sitedfValley[sitedfValley['Basin Type']==geom]['Distance to basin edge'],sitedfValley[sitedfValley['Basin Type']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    # ax[i, j].set_xlim([10,300])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('Distance to basin edge (m)', size=20)
ax[1][1].set_xlabel('Distance to basin edge (m)', size=20)
plt.savefig(os.path.join(out_dir,"ValleyResAnalysisDBE.pdf"))
#%% Residual analysis - Paper 1 - Valley sites ONLY - Vs30
sitedfValley      = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley")]
T0Valley          = stations[(stations['Geomorphology']=="Valley")]["Vs30"]
dS2SValleyUn      = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley")]['stat_id']]
colors           = ['#e6194B','#000075','#3cb44b','#808000']
unique_type      = sitedfValley['Basin Type'].unique()
custom_order     = ['Unmodeled','Type 1','Type 2','Type 3','Type 4']
mask         = T0Valley.notna()
x            = T0Valley[mask]
x            = np.array(x)
y            = dS2SValleyUn
order        = np.argsort(x)
unique_type  = sorted(unique_type, key=lambda x: custom_order.index(x))
IMs              = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_type):
        ax[i,j].scatter(sitedfValley[sitedfValley['Basin Type']==geom]['Vs30'],sitedfValley[sitedfValley['Basin Type']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    # ax[i, j].set_xlim([10,300])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('$V_{S30}$ (m/s)', size=20)
ax[1][1].set_xlabel('$V_{S30}$ (m/s)', size=20)
plt.savefig(os.path.join(out_dir,"ValleyResAnalysisVs30.pdf"))
#%% Residual analysis - Paper 1 - Valley sites ONLY - Z1
sitedfValley      = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley")& (dS2Ssitedf['Basin Type']!="Unmodeled")]
T0Valley          = stations[(stations['Geomorphology']=="Valley") & (stations['Basin Type']!="Unmodeled")]["Z1"]
dS2SValleyUn      = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley")&(ostationsmod['Basin Type'] != "Unmodeled")]['stat_id']]
colors           = ['#000075','#3cb44b','#808000']
unique_type      = sitedfValley['Basin Type'].unique()
custom_order     = ['Type 1','Type 2','Type 3','Type 4']
mask         = T0Valley.notna()
x            = T0Valley[mask]
x            = np.array(x)
y            = dS2SValleyUn
order        = np.argsort(x)
unique_type  = sorted(unique_type, key=lambda x: custom_order.index(x))
IMs              = ['pSA_0.869749002618','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.5)
    for k, geom in enumerate(unique_type):
        ax[i,j].scatter(sitedfValley[sitedfValley['Basin Type']==geom]['Z1'],sitedfValley[sitedfValley['Basin Type']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xlim([10,300])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('$Z_{1.0}$ (m)', size=20)
ax[1][1].set_xlabel('$Z_{1.0}$ (m)', size=20)
plt.savefig(os.path.join(out_dir,"ValleyResAnalysisZ1.pdf"))
#%% Residual analysis - Paper 1 - Basin-edge sites ONLY - T0
sitedfBasinedge      = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Basin-edge")]
T0Basinedge          = stations[(stations['Geomorphology']=="Basin-edge")]["T0"]
dS2SBasinedgeUn      = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Basin-edge")]['stat_id']]
colors           = ['#e6194B','#000075','#3cb44b','#808000']
unique_type      = sitedfBasinedge['Basin Type'].unique()
custom_order     = ['Unmodeled','Type 1','Type 2','Type 3','Type 4']
mask         = T0Basinedge.notna()
x            = T0Basinedge[mask]
x            = np.array(x)
y            = dS2SBasinedgeUn
order        = np.argsort(x)
unique_type  = sorted(unique_type, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_type):
        ax[i,j].scatter(sitedfBasinedge[sitedfBasinedge['Basin Type']==geom]['T0'],sitedfBasinedge[sitedfBasinedge['Basin Type']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xlim([0.1,1.5])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('$T_0$ (s)', size=20)
ax[1][1].set_xlabel('$T_0$ (s)', size=20)
plt.savefig(os.path.join(out_dir,"BasinedgesResAnalysis.pdf"))
#%% Residual analysis - Paper 1 - Basin-edge sites ONLY - Distance to Basin-edge
sitedfBasinedge      = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Basin Edge")& (dS2Ssitedf['Basin Type']!="Unmodeled")]
T0Basinedge          = stations[(stations['Geomorphology']=="Basin-edge")& (stations['Basin Type']!="Unmodeled")]["Distance to basin edge"]
dS2SBasinedgeUn      = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Basin-edge")& (ostationsmod['Basin Type'] != "Unmodeled")]['stat_id']]
colors           = ['#000075','#3cb44b','#808000']
unique_type      = sitedfBasinedge['Basin Type'].unique()
custom_order     = ['Type 1','Type 2','Type 3','Type 4']
mask         = T0Basinedge.notna()
x            = T0Basinedge[mask]
x            = np.array(x)
y            = dS2SBasinedgeUn
order        = np.argsort(x)
unique_type  = sorted(unique_type, key=lambda x: custom_order.index(x))
IMs              = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.4)
    for k, geom in enumerate(unique_type):
        ax[i,j].scatter(sitedfBasinedge[sitedfBasinedge['Basin Type']==geom]['Distance to basin edge'],sitedfBasinedge[sitedfBasinedge['Basin Type']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    # ax[i, j].set_xlim([10,300])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('Distance to basin edge (m)', size=20)
ax[1][1].set_xlabel('Distance to basin edge (m)', size=20)
plt.savefig(os.path.join(out_dir,"BasinedgesResAnalysisDBE.pdf"))
#%% Residual analysis - Paper 1 - Basin-edge sites ONLY - Vs30
sitedfBasinedge      = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Basin-edge")]
T0Basinedge          = stations[(stations['Geomorphology']=="Basin-edge")]["Vs30"]
dS2SBasinedgeUn      = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Basin-edge")]['stat_id']]
colors           = ['#e6194B','#000075','#3cb44b','#808000']
unique_type      = sitedfBasinedge['Basin Type'].unique()
custom_order     = ['Unmodeled','Type 1','Type 2','Type 3','Type 4']
mask         = T0Basinedge.notna()
x            = T0Basinedge[mask]
x            = np.array(x)
y            = dS2SBasinedgeUn
order        = np.argsort(x)
unique_type  = sorted(unique_type, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_type):
        ax[i,j].scatter(sitedfBasinedge[sitedfBasinedge['Basin Type']==geom]['Vs30'],sitedfBasinedge[sitedfBasinedge['Basin Type']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    # ax[i, j].set_xlim([10,300])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('$V_{S30}$ (m/s)', size=20)
ax[1][1].set_xlabel('$V_{S30}$ (m/s)', size=20)
plt.savefig(os.path.join(out_dir,"BasinedgesResAnalysisVs30.pdf"))
#%% Residual analysis - Paper 1 - Basin-edge sites ONLY - Z1
sitedfBasinedge      = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Basin-edge")]
T0Basinedge          = stations[(stations['Geomorphology']=="Basin-edge")]["Z1"]
dS2SBasinedgeUn      = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Basin-edge")]['stat_id']]
colors           = ['#e6194B','#000075','#3cb44b','#808000']
unique_type      = sitedfBasinedge['Basin Type'].unique()
custom_order     = ['Unmodeled','Type 1','Type 2','Type 3','Type 4']
mask         = T0Basinedge.notna()
x            = T0Basinedge[mask]
x            = np.array(x)
y            = dS2SBasinedgeUn
order        = np.argsort(x)
unique_type  = sorted(unique_type, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_type):
        ax[i,j].scatter(sitedfBasinedge[sitedfBasinedge['Basin Type']==geom]['Z1'],sitedfBasinedge[sitedfBasinedge['Basin Type']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    # ax[i, j].set_xlim([10,300])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('$Z_{1.0}$ (m)', size=20)
ax[1][1].set_xlabel('$Z_{1.0}$ (m)', size=20)
plt.savefig(os.path.join(out_dir,"BasinedgesResAnalysisZ1.pdf"))
#%% Plot selected feature and subset four periods - Unmodeled all geom
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']!="Hill") & (dS2Ssitedf['Basin Type']=='Unmodeled Basin')]
T0basinUn    = stations[(stations['Geomorphology']!="Hill") & (stations['Basin Type']=='Unmodeled Basin')]["T0final"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] != "Hill") & (ostationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0final'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    # ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xscale('log')
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledveryshortperiodsgeom.png'))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
# IMs = ['pSA_0.533669923121', 'pSA_0.572236765935', 'pSA_0.613590727341','pSA_0.657933224658']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0final'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    # ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2])
    ax[i, j].set_xscale('log')
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledshortperiods05greatergeom.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
# IMs        = ['pSA_5.722367659350','pSA_7.5','pSA_8.697490026178','pSA_10.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0final'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    # ax[i, j].set_xlim([0,3])
    
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
# plt.savefig(os.path.join(out_dir,'T0Unmodeledlongperiodsgeom.png'))
#%% Plot selected feature and subset four periods - Unmodeled all geom - Old T0
sitedf       = dS2Ssitedf[(dS2Ssitedf['Basin Type']=='Unmodeled Basin')]
T0basinUn    = stations[(stations['Basin Type']=='Unmodeled Basin')]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[ (ostationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
colors       = ['#FF5733','#33FFBD','#9B59B6']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    # ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xscale('log')
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0$ (s)', size=20)
ax[1][1].set_xlabel('$T_0$ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0oldUnmodeledveryshortperiodsgeom.pdf'))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
# IMs = ['pSA_0.533669923121', 'pSA_0.572236765935', 'pSA_0.613590727341','pSA_0.657933224658']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    # ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2])
    ax[i, j].set_xscale('log')
ax[0][0].legend(fontsize=12,frameon=False)
ax[1][0].set_xlabel('$T_0$ (s)', size=20)
ax[1][1].set_xlabel('$T_0$ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0oldUnmodeledshortperiods05greatergeom.pdf'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
# IMs        = ['pSA_5.722367659350','pSA_7.5','pSA_8.697490026178','pSA_10.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    # ax[i, j].set_xlim([0,3])
    
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('$T_0$ (s)', size=20)
ax[1][1].set_xlabel('$T_0$ (s)', size=20)
# plt.savefig(os.path.join(out_dir,'T0oldUnmodeledlongperiodsgeom.pdf'))
#%% Plot selected feature and subset four periods - Unmodeled All Quality T0
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']!="Hill") & (dS2Ssitedf['Basin Type']=='Unmodeled Basin')]
T0basinUn    = stations[(stations['Geomorphology']!="Hill") & (stations['Basin Type']=='Unmodeled Basin')]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] != "Hill") & (ostationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
colors       = ['green','yellow','red']
unique_geom  = sitedf['QT0'].unique()
custom_order = ['Q1','Q2','Q3']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledshortperiodsquality.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledlongperiodsquality.png'))
#%% Plot selected feature and subset four periods - Type 1 All Geom
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']!="Hill") & (dS2Ssitedf['Basin Type']=='Type 1 Basin')]
T0basinUn    = stations[(stations['Geomorphology']!="Hill") & (stations['Basin Type']=='Type 1 Basin')]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] != "Hill") & (ostationsmod['Basin Type'] == 'Type 1 Basin')]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Type 1veryshortperiodsgeom.png'))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Type 1shortperiodsgeom.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Type 1longperiodsgeom.png'))
#%% Plot selected feature and subset four periods - Type 2 All Geom
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']!="Hill") & (dS2Ssitedf['Basin Type']=='Type 2 Basin')]
T0basinUn    = stations[(stations['Geomorphology']!="Hill") & (stations['Basin Type']=='Type 2 Basin')]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] != "Hill") & (ostationsmod['Basin Type'] == 'Type 2 Basin')]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Type 2veryshortperiodsgeom.png'))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Type 2shortperiodsgeom.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Type 2longperiodsgeom.png'))
#%% Plot selected feature and subset four periods - Type 4 All geom
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']!="Hill") & (dS2Ssitedf['Basin Type']=='Type 4 Basin')]
T0basinUn    = stations[(stations['Geomorphology']!="Hill") & (stations['Basin Type']=='Type 4 Basin')]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] != "Hill") & (ostationsmod['Basin Type'] == 'Type 4 Basin')]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Type 4veryshortperiodsgeom.png'))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Type 4shortperiodsgeom.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Type 4longperiodsgeom.png'))
#%% Plot selected feature and subset four periods - Valleys All Quality T0
sitedf       = dS2Ssitedf[dS2Ssitedf['Geomorphology']=="Valley"]
T0basinUn    = stations[stations['Geomorphology']=="Valley"]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[ostationsmod['Geomorphology'] == "Valley"]['stat_id']]
colors       = ['green','yellow','red']
unique_geom  = sitedf['QT0'].unique()
custom_order = ['Q1','Q2','Q3']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Valleyshortperiodsquality.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    # ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xlim([0,3])
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
# plt.savefig(os.path.join(out_dir,'T0Valleylongperiodsquality.png'))
#%% Plot selected feature and subset four periods - Valleys All Quality T0 - Only modelled valleys
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley") & (dS2Ssitedf['Basin Type']!="Unmodeled Basin")]
T0basinUn    = stations[(stations['Geomorphology']=="Valley") & (stations['Basin Type'] !='Unmodeled Basin')]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley") & (ostationsmod['Basin Type'] != "Unmodeled Basin")]['stat_id']]
colors       = ['green','yellow','red']
unique_geom  = sitedf['QT0'].unique()
custom_order = ['Q1','Q2','Q3']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0ModValleyshortperiodsquality.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    # ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range])
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xlim([0,3])
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
# plt.savefig(os.path.join(out_dir,'T0ModValleylongperiodsquality.png'))
#%% Plot selected feature and subset four periods - Valleys All Quality T0 - Only Unmodeled valleys
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley") & (dS2Ssitedf['Basin Type']=="Unmodeled Basin")]
T0basinUn    = stations[(stations['Geomorphology']=="Valley") & (stations['Basin Type'] =='Unmodeled Basin')]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley") & (ostationsmod['Basin Type'] == "Unmodeled Basin")]['stat_id']]
colors       = ['green','yellow','red']
unique_geom  = sitedf['QT0'].unique()
custom_order = ['Q1','Q2','Q3']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0UnModValleyshortperiodsquality.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xlim([0,3])
    # ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
# plt.savefig(os.path.join(out_dir,'T0UnModValleylongperiodsquality.png'))
#%% Plot selected feature and subset four periods - Basin-edges All Quality T0
sitedf       = dS2Ssitedf[dS2Ssitedf['Geomorphology']=="Basin-edge"]
T0basinUn    = stations[stations['Geomorphology']=="Basin-edge"]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[ostationsmod['Geomorphology'] == "Basin-edge"]['stat_id']]
colors       = ['green','yellow','red']
unique_geom  = sitedf['QT0'].unique()
custom_order = ['Q1','Q2','Q3']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Basin-edgeshortperiodsquality.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Basin-edgelongperiodsquality.png'))
#%% Plot selected feature and subset four periods - Basins All Quality T0
sitedf       = dS2Ssitedf[dS2Ssitedf['Geomorphology']=="Basin"]
T0basinUn    = stations[stations['Geomorphology']=="Basin"]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[ostationsmod['Geomorphology'] == "Basin"]['stat_id']]
colors       = ['green','yellow','red']
unique_geom  = sitedf['QT0'].unique()
unique_geom  = unique_geom[~pd.isna(unique_geom)]
custom_order = ['Q1','Q2','Q3']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Basinshortperiodsquality.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Basinlongperiodsquality.png'))
#%% Plot selected feature and subset four periods - ALL NON-HILL SITES
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']!="Hill")]
T0basinUn    = stations[(stations['Geomorphology']!="Hill")]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] != "Hill")]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0final'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    # ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xscale('log')
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledveryshortperiodsgeom.png'))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
# IMs = ['pSA_0.533669923121', 'pSA_0.572236765935', 'pSA_0.613590727341','pSA_0.657933224658']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0final'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    # ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2])
    ax[i, j].set_xscale('log')
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0allbasinsshortperiods05greatergeom.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
# IMs        = ['pSA_5.722367659350','pSA_7.5','pSA_8.697490026178','pSA_10.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0final'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    # ax[i, j].set_xlim([0,3])
    
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0allbasinslongperiodsgeom.png'))
#%% Plot selected feature and subset four periods - ALL MODELED BASIN SITES
sitedf       = dS2Ssitedf[(dS2Ssitedf['Basin Type']=="Type 1 Basin") | (dS2Ssitedf['Basin Type']=='Type 3 Basin') | (dS2Ssitedf['Basin Type']=='Type 4 Basin')]
T0basinUn    = stations[(stations['Basin Type']=="Type 1 Basin") | (stations['Basin Type']=='Type 3 Basin') | (stations['Basin Type']=='Type 4 Basin')]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[(ostationsmod['Basin Type'] == "Type 1 Basin") | (ostationsmod['Basin Type'] == 'Type 3 Basin') | (ostationsmod['Basin Type'] == 'Type 4 Basin')]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0final'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    # ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xscale('log')
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledveryshortperiodsgeom.png'))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
# IMs = ['pSA_0.533669923121', 'pSA_0.572236765935', 'pSA_0.613590727341','pSA_0.657933224658']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0final'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    # ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2])
    ax[i, j].set_xscale('log')
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0modeledbasinsshortperiods05greatergeom.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
# IMs        = ['pSA_5.722367659350','pSA_7.5','pSA_8.697490026178','pSA_10.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0final'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    ax[i,j].set_xscale('log')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
    ax[i, j].set_ylim([-1.2,1.2]) 
    # ax[i, j].set_xlim([0,3])
    
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0modeledbasinslongperiodsgeom.png'))
#%% Plot selected feature and subset four periods - Unmodeled Basin and All remaining Valleys
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley") | (dS2Ssitedf['Basin Type']=='Unmodeled Basin')]
T0basinUn    = stations[(stations['Geomorphology']=="Valley") | (stations['Basin Type']=='Unmodeled Basin')]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley") | (ostationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&valleyshortperiodsgeom.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xlim([0,3]) 
    # ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
# plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&valleylongperiodsgeom.png'))
#%% Plot selected feature and subset four periods - Unmodeled Basin and All remaining Valleys
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley") | (dS2Ssitedf['Basin Type']=='Unmodeled Basin')]
T0basinUn    = stations[(stations['Geomorphology']=="Valley") | (stations['Basin Type']=='Unmodeled Basin')]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley") | (ostationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&valleyshortperiodsgeom.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([-1.2,1.2]) 
    ax[i, j].set_xlim([0,3]) 
    # ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
# plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&valleylongperiodsgeom.png'))
#%% Plot selected feature and subset four periods - quality - Unmodeled Basin and All remaining Valleys
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley") | (dS2Ssitedf['Basin Type']=='Unmodeled Basin')]
T0basinUn    = stations[(stations['Geomorphology']=="Valley") | (stations['Basin Type']=='Unmodeled Basin')]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley") | (ostationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
colors       = ['green','yellow','red']
unique_geom  = sitedf['QT0'].unique()
custom_order = ['Q1','Q2','Q3']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&valleyshortperiodsquality.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&valleylongperiodsquality.png'))
#%% Plot selected feature and subset four periods - Unmodeled Basins and all remaining Valleys and Basin-edges
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley")|(dS2Ssitedf['Geomorphology']=="Basin-edge") | (dS2Ssitedf['Basin Type']=='Unmodeled Basin')]
T0basinUn    = stations[(stations['Geomorphology']=="Valley") | (stations['Geomorphology']=="Basin-edge")| (stations['Basin Type']=='Unmodeled Basin')]["T0"]
dS2SbasinUn = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley") | (ostationsmod['Geomorphology'] == "Basin-edge") | (ostationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&valley&BEshortperiodsgeom.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&valley&BElongperiodsgeom.png'))
#%% Plot selected feature and subset four periods - Quality - Unmodeled Basin and all rem. valleys and all rem. Basin-edges
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley")|(dS2Ssitedf['Geomorphology']=="Basin-edge") | (dS2Ssitedf['Basin Type']=='Unmodeled Basin')]
T0basinUn    = stations[(stations['Geomorphology']=="Valley") | (stations['Geomorphology']=="Basin-edge")| (stations['Basin Type']=='Unmodeled Basin')]["T0"]
dS2SbasinUn = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley") | (ostationsmod['Geomorphology'] == "Basin-edge") | (ostationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
colors       = ['green','yellow','red']
unique_geom  = sitedf['QT0'].unique()
custom_order = ['Q1','Q2','Q3']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&valley&BEshortperiodsquality.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&valley&BElongperiodsquality.png'))
#%% Plot selected feature and subset four periods - Unmodeled Basin and all remaining valleys and all remaining basin-edges and all Wellington basin sites
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley")|(dS2Ssitedf['Geomorphology']=="Basin-edge") | (dS2Ssitedf['Basin Type']=='Unmodeled Basin') | (dS2Ssitedf['Basin Type']=='Type 2 Basin')]
T0basinUn    = stations[(stations['Geomorphology']=="Valley") | (stations['Geomorphology']=="Basin-edge")| (stations['Basin Type']=='Unmodeled Basin') | (stations['Basin Type']=='Type 2 Basin')]["T0"]
dS2SbasinUn = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley") | (ostationsmod['Geomorphology'] == "Basin-edge") | (ostationsmod['Basin Type'] == 'Unmodeled Basin') | (ostationsmod['Basin Type'] == 'Type 2 Basin')]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&Type2&valley&BEshortperiodsgeom.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&Type2&valley&BElongperiodsgeom.png'))
#%% Plot selected feature and subset four periods - Quality - Unmodeled Basin and all remaining Basin-edges and all remaining valleys and all Welli sites
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley")|(dS2Ssitedf['Geomorphology']=="Basin-edge") | (dS2Ssitedf['Basin Type']=='Unmodeled Basin') | (dS2Ssitedf['Basin Type']=='Type 2 Basin')]
T0basinUn    = stations[(stations['Geomorphology']=="Valley") | (stations['Geomorphology']=="Basin-edge")| (stations['Basin Type']=='Unmodeled Basin') | (stations['Basin Type']=='Type 2 Basin')]["T0"]
dS2SbasinUn = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley") | (ostationsmod['Geomorphology'] == "Basin-edge") | (ostationsmod['Basin Type'] == 'Unmodeled Basin') | (ostationsmod['Basin Type'] == 'Type 2 Basin')]['stat_id']]
colors       = ['green','yellow','red']
unique_geom  = sitedf['QT0'].unique()
unique_geom  = unique_geom[~pd.isna(unique_geom)]
custom_order = ['Q1','Q2','Q3']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&Type2&valley&BEshortperiodsquality.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['QT0']==geom]['T0'],sitedf[sitedf['QT0']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('$T_0 (database) $ (s)', size=20)
ax[1][1].set_xlabel('$T_0 (database) $ (s)', size=20)
plt.savefig(os.path.join(out_dir,'T0Unmodeledbasins&Type2&valley&BElongperiodsquality.png'))
#%% Plot selected feature and subset four periods - Only BE - Slope
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Basin-edge")]
T0basinUn    = stations[ (stations['Geomorphology']=="Basin-edge")]["Slope"]
dS2SbasinUn = dS2Sres_all.loc[ostationsmod[ (ostationsmod['Geomorphology'] == "Basin-edge")]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['Slope'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('Slope (m/m)', size=20)
ax[1][1].set_xlabel('Slope (m/m)', size=20)
plt.savefig(os.path.join(out_dir,'SlopeBEshortperiodsgeom.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['Slope'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('Slope (m/m)', size=20)
ax[1][1].set_xlabel('Slope (m/m)', size=20)
plt.savefig(os.path.join(out_dir,'SlopeBElongperiodsgeom.png'))
#%% Plot selected feature and subset four periods - BE & Valley Slopes
sitedf       = dS2Ssitedf[(dS2Ssitedf['Geomorphology']=="Valley")|(dS2Ssitedf['Geomorphology']=="Basin-edge")]
T0basinUn    = stations[(stations['Geomorphology']=="Valley") | (stations['Geomorphology']=="Basin-edge")]["Slope"]
dS2SbasinUn = dS2Sres_all.loc[ostationsmod[(ostationsmod['Geomorphology'] == "Valley") | (ostationsmod['Geomorphology'] == "Basin-edge")]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin-edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['Slope'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('Slope (m/m)', size=20)
ax[1][1].set_xlabel('Slope (m/m)', size=20)
plt.savefig(os.path.join(out_dir,'Slopevalley&BEshortperiodsgeom.png'))
IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['Slope'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], s=40,edgecolor='k',label=geom)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2)
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='gray', alpha=0.3)
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    y_min, y_max = ax[i,j].get_ylim()
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range]) 
ax[0][0].legend(fontsize=12)
ax[1][0].set_xlabel('Slope (m/m)', size=20)
ax[1][1].set_xlabel('Slope (m/m)', size=20)
plt.savefig(os.path.join(out_dir,'Slopevalley&BElongperiodsgeom.png'))
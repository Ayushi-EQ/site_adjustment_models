# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:27:35 2024

@author: ati47
"""

import numpy as np
import pandas as pd
import addcopyfighandler
from pathlib import Path
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import scipy.stats as stats
data_dir1           = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\New Results")
ObsIMs              = pd.read_csv(Path(data_dir1 / "im_obs.csv"))
SimIMs              = pd.read_csv(data_dir1 / "im_sim.csv")
dS2Sres_all         = pd.read_csv(data_dir1/ "Residuals"/ "PJSreStationBiased_sim.csv",index_col=0)
dS2Serr_all         = pd.read_csv(data_dir1/"Residuals"/ "PJSreStationerror_sim.csv",index_col=0)
varcomps_all        = pd.read_csv(data_dir1/"Residuals"/ "PJSvarCompsBiased_sim.csv",index_col=0)
Features            = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\9. Model development\Correlation\Features.csv"),index_col = 'Station Code')
Features.index      = 'Station_' + Features.index.astype(str)
stations            = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\stations.csv"))
stations            = pd.merge(stations[['stat_id','stat_name']],Features[['stat_name','Basin Type','Slope','Geomorphology','T0']],on='stat_name')
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
out_dir             = Path(os.path.join(os.getcwd(),"Unmodelled basins","Final Basin Model_Draft 2"))
#%% LOWESS function define
def loess(x, y, f):
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
#%% Slope changes
sitedf       = dS2Ssitedf[(dS2Ssitedf['Basin Type']=='Unmodeled')]
T0basinUn    = stations[ (stations['Basin Type']=='Unmodeled Basin')]["T0"]
dS2SbasinUn  = dS2Sres_all.loc[ostationsmod[ (ostationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
colors       = ['red','orange','blue']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin Edge','Valley']
mask         = T0basinUn.notna()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
order        = np.argsort(x)
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
#%%
# Create a dictionary to store results with period as the key
results = {'Period': [], 'Steepness Change': [], 'Slope Change': []}

for idx, im in enumerate(dS2Sres_all.columns):
    dS2S_sel = y.loc[:, im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    lowessold, lowessoldstd = loess(x, np.array(dS2S_sel), 0.3)
    _, unique_indices = np.unique(x[order], return_index=True)
    unique_x_order = x[order][sorted(unique_indices)]
    unique_lowessold = lowessold[order][sorted(unique_indices)]
    smoothed_slope = np.gradient(unique_lowessold, unique_x_order)
    steepness_threshold = 0.20  # Adjust based on visual tuning
    steepness_change_idx = np.where((smoothed_slope[:-1] > steepness_threshold) & (smoothed_slope[1:] <= steepness_threshold))[0]
    
    # Detect the direction change point (actual slope sign change)
    direction_change_idx = np.where((smoothed_slope[:-1] > 0) & (smoothed_slope[1:] <= 0))[0]
    
    # Store steepness and direction change points
    results['Period'].append(im)
    results['Steepness Change'].append(unique_x_order[steepness_change_idx[0]] if len(steepness_change_idx) > 0 else np.nan)
    results['Slope Change'].append(unique_x_order[direction_change_idx[0]] if len(direction_change_idx) > 0 else np.nan)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
results_df['Numeric Period'] = results_df['Period'].str.extract(r'(\d+\.\d+)').astype(float)
filtered_df1 = results_df[(results_df['Numeric Period'] > 1) & (results_df['Numeric Period']<5)]
filtered_df2 = results_df[(results_df['Numeric Period'] <= 1) & (results_df['Numeric Period'] >= 0.5)]
filtered_df3 = results_df[results_df['Numeric Period'] >= 5]

steepness_change_avg1 = filtered_df1['Steepness Change'].mean()
steepness_change_avg2 = filtered_df2['Steepness Change'].mean()
steepness_change_avg3 = filtered_df3['Steepness Change'].mean()
fig,ax = plt.subplots(constrained_layout=True)
ax.semilogx(T_RS.values,results_df['Steepness Change'].values,color='r',label='Steepness change')
# ax.semilogx(T_RS.values,results_df['Slope Change'].values,color='b',label='Slope change')
ax.axhline(steepness_change_avg1,color = 'k')
ax.axhline(steepness_change_avg2,color = 'k',linestyle='--')
ax.axhline(steepness_change_avg3,color = 'k',linestyle='-.')
ax.legend(fontsize=14,frameon=False)
ax.set_xlabel("Vibration Period",fontsize=16)
ax.set_ylabel("Period at which change happened")
plt.savefig(os.path.join(out_dir,"Period Vs. steepness change.png"))
#%% Basin Model MFinal
def TiwariMFinalUn(Period,T,m1,intercept,c_low,c_high,thresh1,thresh2,thresh3,thresh4,thresh11):
    if (Period >= 0.5) & (Period <=1):
        return np.piecewise(
            T,
            [T <=thresh11,
              (T > thresh11) & (T <= thresh2),
              T>thresh2,
              ],
            [lambda T: c_low,
              lambda T: m1*np.log(T) + intercept,
              lambda T: c_high
              ]
            )
    elif (Period>1) & (Period<5):
        return np.piecewise(
            T,
            [T <=thresh11,
              (T > thresh11) & (T <= thresh3),
              T>thresh3,
              ],
            [lambda T: c_low,
              lambda T: m1*np.log(T) + intercept,
              lambda T: c_high
              ]
            )
    elif Period>=5:
        return np.piecewise(
            T,
            [T <=thresh11,
              (T > thresh11) & (T <= thresh4),
              T>thresh4,
              ],
            [lambda T: c_low,
              lambda T: m1*np.log(T) + intercept,
              lambda T: c_high
              ]
            )
    else:
        return np.piecewise(
            T,
            [T <=thresh2,
              T>thresh2,
              ],
            [ lambda T: m1*np.log(T) + intercept,
              lambda T: c_high
              ]
            )
#%% Slope calculation - bootstrapping trial
n_bootstrap = 1000
mask       = T0basinUn.notna()
x          = T0basinUn[mask]
y          = dS2SbasinUn
# IMs        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4']  
# IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
IMs        = ['pSA_0.869749002618','pSA_2.0','pSA_3.0','pSA_5.0']   
# IMs        = ['pSA_6.135907273413','pSA_7.5','pSA_8.697490026178','pSA_10.0']
thresh1 = 0.3
thresh11 = 0.4
thresh2 = 0.95
thresh3 = 1.1
thresh4 = 1.1
T0model = np.append(T0basinUn,[thresh1,thresh2,thresh3,thresh4])
fig,ax  = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    SitedS2SDB = pd.DataFrame({'T0Un': x, 'dS2S': dS2S_sel.values})
    m1_dist = []
    c_high_dist = []
    c_low_dist  = []
    inter_dist = []
    Period = float(im.split('_')[1])
    for _ in range(n_bootstrap):
        SitedS2SDB_bootstrap = SitedS2SDB.sample(frac=1, replace=True)
        if (Period >= 0.5) & (Period <=1):
            mask1 = (SitedS2SDB_bootstrap['T0Un'].values <= thresh2) & (SitedS2SDB_bootstrap['T0Un'].values > thresh1)
            filtered_T01   = SitedS2SDB_bootstrap['T0Un'][mask1]
            filtered_olddS2SUn1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
            m1, intercept,r,p,std = linregress(np.log(filtered_T01), filtered_olddS2SUn1)
            c_high = m1*np.log(thresh2) + intercept
            c_low  = m1*np.log(thresh11) + intercept
        elif (Period>1) & (Period<5):
            mask1 = (SitedS2SDB_bootstrap['T0Un'].values <= thresh3) & (SitedS2SDB_bootstrap['T0Un'].values > thresh1)
            filtered_T01   = SitedS2SDB_bootstrap['T0Un'][mask1]
            filtered_olddS2SUn1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
            m1, intercept,r,p,std = linregress(np.log(filtered_T01), filtered_olddS2SUn1)
            c_high = m1*np.log(thresh3) + intercept
            c_low  = m1*np.log(thresh11) + intercept
        elif Period>=5:
            mask1 = (SitedS2SDB_bootstrap['T0Un'].values <= thresh4) & (SitedS2SDB_bootstrap['T0Un'].values > thresh1)
            filtered_T01   = SitedS2SDB_bootstrap['T0Un'][mask1]
            filtered_olddS2SUn1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
            m1, intercept,r,p,std = linregress(np.log(filtered_T01), filtered_olddS2SUn1)
            c_high = m1*np.log(thresh4) + intercept
            c_low  = m1*np.log(thresh11) + intercept
        else:
            mask1 = SitedS2SDB_bootstrap['T0Un'].values <= thresh2
            filtered_T01   = SitedS2SDB_bootstrap['T0Un'][mask1]
            filtered_olddS2SUn1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
            m1, intercept,r,p,std = linregress(np.log(filtered_T01), filtered_olddS2SUn1)
            c_high = m1*np.log(thresh2) + intercept
            c_low  = np.nan
        m1_dist.append(m1)
        c_high_dist.append(c_high)
        c_low_dist.append(c_low)
        inter_dist.append(intercept)
        ans = TiwariMFinalUn(Period,T0model, m1,intercept,c_low,c_high,thresh1,thresh2,thresh3,thresh4,thresh11)
        df = pd.DataFrame({'T0Un': T0model, 'dS2S': ans})
        df = df.sort_values(by='T0Un')
        ax[i, j].plot(df['T0Un'],df['dS2S'],'gray',alpha=0.05)
    ax[i,j].scatter(x, np.array(dS2S_sel), color='#e6194B',edgecolor='k')
    HMdist   = pd.DataFrame({'clow': c_low_dist,'chigh':c_high_dist,'m1':m1_dist,'intercept':inter_dist})
    m1_mean    = HMdist['m1'].mean() 
    chigh_mean = HMdist['chigh'].mean()
    clow_mean  = HMdist['clow'].mean()
    inter_mean = HMdist['intercept'].mean()
    m1_std     = HMdist['m1'].std()
    chigh_std  = HMdist['chigh'].std()
    clow_std   = HMdist['clow'].std()
    inter_std  = HMdist['intercept'].std()
    ans = TiwariMFinalUn(Period,T0model, m1_mean, inter_mean,clow_mean,chigh_mean,thresh1,thresh2,thresh3,thresh4,thresh11)
    df = pd.DataFrame({'T0Un': T0model, 'dS2S': ans})
    df = df.sort_values(by='T0Un')
    ax[i, j].plot(df['T0Un'], df['dS2S'], 'k')
    ax[i, j].axhline(0, color='k', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    if i == 0 and j == 0:
        ax[i, j].axvline(thresh2, color='maroon', linestyle='--')
    else:
        ax[i, j].axvline(thresh3, color='maroon', linestyle='--')
    ax[i, j].axvline(thresh11, color='maroon', linestyle='--')
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    ax[i,j].set_xscale('log')
    

global_y_min, global_y_max = float('inf'), float('-inf')
for i in range(2):  # rows and cols define the subplot grid
    for j in range(2):
        y_min, y_max = ax[i, j].get_ylim()
        global_y_min = min(global_y_min, y_min)
        global_y_max = max(global_y_max, y_max)

y0= 0
max_y_range = max(abs(global_y_max - y0), abs(global_y_min - y0))
rounded_value = np.round(float(im.split('pSA_')[1]), 2)
for i in range(2):
    for j in range(2):
        ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range])
ax[1][0].set_xlabel('$T_0$ (s)', size=20)
ax[1][1].set_xlabel('$T_0$ (s)', size=20)

# plt.savefig(os.path.join(out_dir,"BootstrapslopeSampleVeryShort.png"))
# plt.savefig(os.path.join(out_dir,"BootstrapslopeSampleShort.png"))
plt.savefig(os.path.join(out_dir,"BasinBootstrapslopeSampleLong.pdf"))
# plt.savefig(os.path.join(out_dir,"BootstrapslopeSampleVeryLong.png"))
#%% Slope calculation - Model and Loess
# colors       = ['b','#006400','#4b0082']
colors = ['#1E90FF', '#66C266', '#9370DB', '#D17878']
unique_geom  = sitedf['Geomorphology'].unique()
custom_order = ['Basin','Basin Edge','Valley']
unique_geom  = sorted(unique_geom, key=lambda x: custom_order.index(x))
n_bootstrap = 1000
mask       = T0basinUn.notna()
x          = T0basinUn[mask]
x          = np.array(x)
y          = dS2SbasinUn
# IMs        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4']  
# IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
IMs        = ['pSA_0.869749002618','pSA_2.0','pSA_3.0','pSA_5.0']    
# IMs        = ['pSA_6.135907273413','pSA_7.5','pSA_8.697490026178','pSA_10.0']
thresh1 = 0.3
thresh11 = 0.4
thresh2 = 0.95
thresh3 = 1.1
thresh4 = 1.1
alpha=1
T0model = np.append(T0basinUn,[thresh1,thresh2,thresh3,thresh4])
fig,ax  = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    SitedS2SDB = pd.DataFrame({'T0Un': x, 'dS2S': dS2S_sel.values})
    m1_dist = []
    c_high_dist = []
    c_low_dist  = []
    inter_dist = []
    Period = float(im.split('_')[1])
    bootstrap_curves = []
    for _ in range(n_bootstrap):
        SitedS2SDB_bootstrap = SitedS2SDB.sample(frac=1, replace=True)
        if (Period >= 0.5) & (Period <=1):
            mask1 = (SitedS2SDB_bootstrap['T0Un'].values <= thresh2) & (SitedS2SDB_bootstrap['T0Un'].values > thresh1)
            filtered_T01   = SitedS2SDB_bootstrap['T0Un'][mask1]
            filtered_olddS2SUn1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
            m1, intercept,r,p,std = linregress(np.log(filtered_T01), filtered_olddS2SUn1)
            c_high = m1*np.log(thresh2) + intercept
            c_low  = m1*np.log(thresh11) + intercept
        elif (Period>1) & (Period<5):
            mask1 = (SitedS2SDB_bootstrap['T0Un'].values <= thresh3) & (SitedS2SDB_bootstrap['T0Un'].values > thresh1)
            filtered_T01   = SitedS2SDB_bootstrap['T0Un'][mask1]
            filtered_olddS2SUn1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
            m1, intercept,r,p,std = linregress(np.log(filtered_T01), filtered_olddS2SUn1)
            c_high = m1*np.log(thresh3) + intercept
            c_low  = m1*np.log(thresh11) + intercept
        elif Period>=5:
            mask1 = (SitedS2SDB_bootstrap['T0Un'].values <= thresh4) & (SitedS2SDB_bootstrap['T0Un'].values > thresh1)
            filtered_T01   = SitedS2SDB_bootstrap['T0Un'][mask1]
            filtered_olddS2SUn1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
            m1, intercept,r,p,std = linregress(np.log(filtered_T01), filtered_olddS2SUn1)
            c_high = m1*np.log(thresh4) + intercept
            c_low  = m1*np.log(thresh11) + intercept
        else:
            mask1 = (SitedS2SDB_bootstrap['T0Un'].values <= thresh2)
            filtered_T01   = SitedS2SDB_bootstrap['T0Un'][mask1]
            filtered_olddS2SUn1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
            m1, intercept,r,p,std = linregress(np.log(filtered_T01), filtered_olddS2SUn1)
            c_high = m1*np.log(thresh2) + intercept
            c_low  = np.nan
        m1_dist.append(m1)
        c_high_dist.append(c_high)
        c_low_dist.append(c_low)
        inter_dist.append(intercept)
        ans = TiwariMFinalUn(Period,T0model, m1,intercept,c_low,c_high,thresh1,thresh2,thresh3,thresh4,thresh11)
        df = pd.DataFrame({'T0Un': T0model, 'dS2S': ans})
        df = df.sort_values(by='T0Un')
        # ax[i, j].plot(df['T0Un'],df['dS2S'],'gray',alpha=0.05)
        bootstrap_curves.append(ans)
    # ax[i,j].scatter(x, np.array(dS2S_sel), color='#e6194B',edgecolor='k')
    for k, geom in enumerate(unique_geom):
        ax[i,j].scatter(sitedf[sitedf['Geomorphology']==geom]['T0'],sitedf[sitedf['Geomorphology']==geom][im],color=colors[k], alpha=alpha,s=40,edgecolor='k',label=geom)
    
    bootstrap_curves = np.array(bootstrap_curves)
    mean_ans = np.mean(bootstrap_curves, axis=0)
    lower_ci = np.percentile(bootstrap_curves, 2.5, axis=0)
    upper_ci = np.percentile(bootstrap_curves, 97.5, axis=0)
    
    
    
    HMdist   = pd.DataFrame({'clow': c_low_dist,'chigh':c_high_dist,'m1':m1_dist,'intercept':inter_dist})
    m1_mean    = HMdist['m1'].mean() 
    chigh_mean = HMdist['chigh'].mean()
    clow_mean  = HMdist['clow'].mean()
    inter_mean = HMdist['intercept'].mean()
    m1_std     = HMdist['m1'].std()
    chigh_std  = HMdist['chigh'].std()
    clow_std   = HMdist['clow'].std()
    inter_std  = HMdist['intercept'].std()
    ans = TiwariMFinalUn(Period,T0model, m1_mean, inter_mean,clow_mean,chigh_mean,thresh1,thresh2,thresh3,thresh4,thresh11)
    df = pd.DataFrame({'T0Un': T0model, 'dS2S': ans})
    df = df.sort_values(by='T0Un')
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    ax[i,j].plot(x[order], lowessold[order], color='k', linewidth=2,label='LOWESS')
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='k', alpha=0.1)
    ax[i, j].plot(df['T0Un'], df['dS2S'], '#FF5733',label='Model',linewidth=2)
    ax[i, j].fill_between(df['T0Un'], lower_ci[df.index], upper_ci[df.index], color='#FF5733', alpha=0.15)
    ax[i, j].axhline(0, color='k', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    if i == 0 and j == 0:
        ax[i, j].axvline(thresh2, color='maroon', linestyle='--')
    else:
        ax[i, j].axvline(thresh3, color='maroon', linestyle='--')
    ax[i, j].axvline(thresh11, color='maroon', linestyle='--')
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    ax[i,j].set_xscale('log')

global_y_min, global_y_max = float('inf'), float('-inf')
for i in range(2):  # rows and cols define the subplot grid
    for j in range(2):
        y_min, y_max = ax[i, j].get_ylim()
        global_y_min = min(global_y_min, y_min)
        global_y_max = max(global_y_max, y_max)

y0= 0
max_y_range = max(abs(global_y_max - y0), abs(global_y_min - y0))
rounded_value = np.round(float(im.split('pSA_')[1]), 2)
for i in range(2):
    for j in range(2):
        ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range])
# ax[0][0].legend(fontsize=14)
# Get all handles and labels
handles, labels = ax[0, 0].get_legend_handles_labels()

# Separate the two groups
geom_handles = [h for h, l in zip(handles, labels) if l in custom_order]
geom_labels  = [l for l in labels if l in custom_order]

line_handles = [h for h, l in zip(handles, labels) if l in ['LOWESS', 'Model']]
line_labels  = [l for l in labels if l in ['LOWESS', 'Model']]

# First legend: Geomorphology (left side)
legend1 = ax[0, 0].legend(geom_handles, geom_labels, loc='upper left', fontsize=16)

# Second legend: Fit types (right side)
legend2 = ax[0, 0].legend(line_handles, line_labels, loc='lower right', fontsize=16)

# Add first legend back after second replaces it
ax[0, 0].add_artist(legend1)

ax[1][0].set_xlabel('$T_0$ (s)', size=20)
ax[1][1].set_xlabel('$T_0$ (s)', size=20)
ax[1][0].tick_params(axis='x', pad=10)
ax[1][1].tick_params(axis='x', pad=10)
# plt.savefig(os.path.join(out_dir,"BootstrapslopeLoessVeryShort.png"))
# plt.savefig(os.path.join(out_dir,"BootstrapslopeLoessShort.png"))
plt.savefig(os.path.join(out_dir,"BootstrapslopeLoessLong.pdf"))
# plt.savefig(os.path.join(out_dir,"BootstrapslopeLoessVeryLong.png"))
#%% Bootstrapping for all periods
n_bootstrap = 1000
mask       = T0basinUn.notna()
x          = T0basinUn[mask]
y          = dS2SbasinUn
summary_stats = []
thresh1 = 0.3
thresh2 = 0.95
thresh11 = 0.4
thresh3 = 1.1
thresh4 = 1.1
T0model = np.append(T0basinUn,[thresh1,thresh2,thresh3,thresh4])
for im in dS2Sres_all.columns:
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    SitedS2SDB = pd.DataFrame({'T0Un': x, 'dS2S': dS2S_sel.values})
    m1_dist = []
    c_high_dist = []
    c_low_dist  = []
    inter_dist = []
    p_dist = []
    p_interceptdist = []
    Period = float(im.split('_')[1])
    for _ in range(n_bootstrap):
        SitedS2SDB_bootstrap = SitedS2SDB.sample(frac=1, replace=True)
        if (Period >= 0.5) & (Period <=1):
            mask1 = (SitedS2SDB_bootstrap['T0Un'].values <= thresh2) & (SitedS2SDB_bootstrap['T0Un'].values > thresh1)
            filtered_T01   = SitedS2SDB_bootstrap['T0Un'][mask1]
            filtered_olddS2SUn1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
            m1, intercept,r,p,std = linregress(np.log(filtered_T01), filtered_olddS2SUn1)
            c_high = m1*np.log(thresh2) + intercept
            c_low  = m1*np.log(thresh11) + intercept
            x_mean = np.mean(filtered_T01)
            n = len(filtered_T01)
            intercept_std_err = std * np.sqrt(np.sum(filtered_T01**2) / (n * np.sum((filtered_T01 - x_mean)**2)))
            t_stat_intercept = intercept / intercept_std_err
            p_value_intercept = 2 * (1 - stats.t.cdf(np.abs(t_stat_intercept), df=n - 2))
        elif (Period>1) & (Period<5):
            mask1 = (SitedS2SDB_bootstrap['T0Un'].values <= thresh3) & (SitedS2SDB_bootstrap['T0Un'].values > thresh1)
            filtered_T01   = SitedS2SDB_bootstrap['T0Un'][mask1]
            filtered_olddS2SUn1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
            m1, intercept,r,p,std = linregress(np.log(filtered_T01), filtered_olddS2SUn1)
            c_high = m1*np.log(thresh3) + intercept
            c_low  = m1*np.log(thresh11) + intercept
            x_mean = np.mean(filtered_T01)
            n = len(filtered_T01)
            intercept_std_err = std * np.sqrt(np.sum(filtered_T01**2) / (n * np.sum((filtered_T01 - x_mean)**2)))
            t_stat_intercept = intercept / intercept_std_err
            p_value_intercept = 2 * (1 - stats.t.cdf(np.abs(t_stat_intercept), df=n - 2))
        elif Period>=5:
            mask1 = (SitedS2SDB_bootstrap['T0Un'].values <= thresh4) & (SitedS2SDB_bootstrap['T0Un'].values > thresh1)
            filtered_T01   = SitedS2SDB_bootstrap['T0Un'][mask1]
            filtered_olddS2SUn1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
            m1, intercept,r,p,std = linregress(np.log(filtered_T01), filtered_olddS2SUn1)
            c_high = m1*np.log(thresh4) + intercept
            c_low  = m1*np.log(thresh11) + intercept
            x_mean = np.mean(filtered_T01)
            n = len(filtered_T01)
            intercept_std_err = std * np.sqrt(np.sum(filtered_T01**2) / (n * np.sum((filtered_T01 - x_mean)**2)))
            t_stat_intercept = intercept / intercept_std_err
            p_value_intercept = 2 * (1 - stats.t.cdf(np.abs(t_stat_intercept), df=n - 2))
        else:
            mask1 = SitedS2SDB_bootstrap['T0Un'].values <= thresh2
            filtered_T01   = SitedS2SDB_bootstrap['T0Un'][mask1]
            filtered_olddS2SUn1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
            m1, intercept,r,p,std = linregress(np.log(filtered_T01), filtered_olddS2SUn1)
            c_high = m1*np.log(thresh2) + intercept
            c_low  = np.nan
            x_mean = np.mean(filtered_T01)
            n = len(filtered_T01)
            intercept_std_err = std * np.sqrt(np.sum(filtered_T01**2) / (n * np.sum((filtered_T01 - x_mean)**2)))
            t_stat_intercept = intercept / intercept_std_err
            p_value_intercept = 2 * (1 - stats.t.cdf(np.abs(t_stat_intercept), df=n - 2))
        p_dist.append(p)
        p_interceptdist.append(p_value_intercept)
        m1_dist.append(m1)
        c_high_dist.append(c_high)
        c_low_dist.append(c_low)
        inter_dist.append(intercept)
        ans = TiwariMFinalUn(Period,T0model, m1,intercept,c_low,c_high,thresh1,thresh2,thresh3,thresh4,thresh11)
        df = pd.DataFrame({'T0Un': T0model, 'dS2S': ans})
        df = df.sort_values(by='T0Un')

    HMdist   = pd.DataFrame({'clow': c_low_dist,'chigh':c_high_dist,'m1':m1_dist,'intercept':inter_dist,'p-value':p_dist,'p-valueint': p_interceptdist})
    m1_mean    = HMdist['m1'].mean() 
    chigh_mean = HMdist['chigh'].mean()
    clow_mean  = HMdist['clow'].mean()
    inter_mean = HMdist['intercept'].mean()
    p_mean     = HMdist['p-value'].mean()
    p_intmean  = HMdist['p-valueint'].mean()
    m1_std     = HMdist['m1'].std()
    chigh_std  = HMdist['chigh'].std()
    clow_std   = HMdist['clow'].std()
    inter_std  = HMdist['intercept'].std()
    summary_stats.append({
        'IM_sel': im,
        'm1_mean': m1_mean,
        'chigh_mean': chigh_mean,
        'clow_mean': clow_mean,
        'intercept_mean': inter_mean,
        'p_mean': p_mean,
        'p_intmean':p_intmean,
        'm1_std':m1_std,
        'chigh_std': chigh_std,
        'inter_std': inter_std
    })
summary_df = pd.DataFrame(summary_stats)
summary_df.set_index('IM_sel', inplace=True)
summary_df.to_excel(os.path.join(out_dir,"Bootstrapping_results.xlsx"))
#%% Plotting model
summary_df = pd.read_excel(os.path.join(out_dir,"Bootstrapping_results.xlsx"),index_col=0)
fig,ax = plt.subplots(2,1,figsize=(7.76, 6.48),constrained_layout=True,sharex=True)
ax[0].semilogx(T_RS.values,summary_df['m1_mean'],'r')
ax[0].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[0].set_ylabel('Proposed slope',size=16)
ax[0].set_xlim([0.01,10])
ax[1].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[1].set_xlabel('Vibration Period (s)',size=16)
ax[1].set_ylabel('p-value of proposed slope',size=16)
ax[1].semilogx(T_RS.values,summary_df['p_mean'],'b')
ax[1].axhline(0.05,color='red',linestyle='--')
plt.savefig(os.path.join(out_dir,"Proposed slope params.png"))
fig,ax = plt.subplots(2,1,figsize=(7.76, 6.48),constrained_layout=True,sharex=True)
ax[0].semilogx(T_RS.values,summary_df['intercept_mean'],'r')
ax[0].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[0].set_ylabel('Proposed intercept',size=16)
ax[0].set_xlim([0.01,10])
ax[1].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[1].set_xlabel('Vibration Period (s)',size=16)
ax[1].set_ylabel('p-value of proposed intercept',size=16)
ax[1].semilogx(T_RS.values,summary_df['p_intmean'],'b')
ax[1].axhline(0.05,color='red',linestyle='--')
plt.savefig(os.path.join(out_dir,"Proposed intercept params.png"))
fig,ax = plt.subplots(2,1,figsize=(7.76, 6.48),constrained_layout=True,sharex=True)
ax[0].semilogx(T_RS.values,summary_df['chigh_mean'],'r')
ax[0].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[0].set_ylabel('Proposed high offset',size=16)
ax[0].set_xlim([0.01,10])
ax[1].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[1].set_xlabel('Vibration Period (s)',size=16)
ax[1].set_ylabel('Proposed low offset',size=16)
ax[1].semilogx(T_RS.values,summary_df['clow_mean'],'b')
plt.savefig(os.path.join(out_dir,"Proposed other model params.png"))
#%% Smoothing the model & fixing p-values
from scipy.interpolate import interp1d
window_size=10
smooth_summary_df = summary_df.copy()
smooth_summary_df = smooth_summary_df.drop(['p_intmean','m1_std','chigh_std','inter_std'],axis=1)
smooth_summary_df['m1_smoothed'] = smooth_summary_df['m1_mean'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')  
smooth_summary_df['inter_smoothed'] = smooth_summary_df['intercept_mean'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')  
smooth_summary_df['T'] = T_RS
transition_period = smooth_summary_df[smooth_summary_df['p_mean'] < 0.05]['T'].iloc[0]
start_value = 0.5
smooth_summary_df.loc[smooth_summary_df['T'] < transition_period, 'm1_smoothed'] = 0
smooth_summary_df.loc[smooth_summary_df['T'] < transition_period, 'inter_smoothed'] = 0
transition_T_values = smooth_summary_df[(smooth_summary_df['T'] >= start_value) & (smooth_summary_df['T'] <= transition_period)]['T']
start_slope_value = 0  # starting slope value for the transition
end_slope_value = smooth_summary_df['m1_smoothed'][smooth_summary_df['T'] > transition_period].iloc[0]
end_slope_valueinter = smooth_summary_df['inter_smoothed'][smooth_summary_df['T'] > transition_period].iloc[0]
interpolator = interp1d(
    [start_value, transition_period],
    [start_slope_value, end_slope_value],
    fill_value="extrapolate"
)
interpolatorinter = interp1d(
    [start_value, transition_period],
    [start_slope_value, end_slope_valueinter],
    fill_value="extrapolate"
)
smooth_summary_df.loc[(smooth_summary_df['T'] >= start_value) & (smooth_summary_df['T'] <= transition_period), 'm1_smoothed'] = interpolator(transition_T_values)
smooth_summary_df.loc[(smooth_summary_df['T'] >= start_value) & (smooth_summary_df['T'] <= transition_period), 'inter_smoothed'] = interpolatorinter(transition_T_values)
smooth_summary_df['chigh_smoothed'] = np.nan
smooth_summary_df['clow_smoothed'] = np.nan
smooth_summary_df.loc[(smooth_summary_df['T'] >= 0.5) & (smooth_summary_df['T'] <= 1), 'chigh_smoothed'] = (np.log(thresh2) * smooth_summary_df['m1_smoothed'] + smooth_summary_df['inter_smoothed'])
smooth_summary_df.loc[(smooth_summary_df['T'] >= 0.5) & (smooth_summary_df['T'] <= 1), 'clow_smoothed']  = (np.log(thresh11) * smooth_summary_df['m1_smoothed'] + smooth_summary_df['inter_smoothed'])
smooth_summary_df.loc[(smooth_summary_df['T'] > 1) & (smooth_summary_df['T'] < 5), 'chigh_smoothed']     = (np.log(thresh3) * smooth_summary_df['m1_smoothed'] + smooth_summary_df['inter_smoothed'])
smooth_summary_df.loc[(smooth_summary_df['T'] > 1) & (smooth_summary_df['T'] < 5), 'clow_smoothed']      = (np.log(thresh11) * smooth_summary_df['m1_smoothed'] + smooth_summary_df['inter_smoothed'])
smooth_summary_df.loc[smooth_summary_df['T'] >= 5, 'chigh_smoothed']                                     = (np.log(thresh4) * smooth_summary_df['m1_smoothed'] + smooth_summary_df['inter_smoothed'])
smooth_summary_df.loc[smooth_summary_df['T'] >= 5, 'clow_smoothed']                                      = (np.log(thresh11) * smooth_summary_df['m1_smoothed'] + smooth_summary_df['inter_smoothed'])
smooth_summary_df.loc[~((smooth_summary_df['T'] >= 0.5) & (smooth_summary_df['T'] <= 1)) &
                      ~((smooth_summary_df['T'] > 1) & (smooth_summary_df['T'] < 5)) &
                      ~(smooth_summary_df['T'] >= 5), 'chigh_smoothed'] = (
    np.log(thresh2) * smooth_summary_df['m1_smoothed'] + smooth_summary_df['inter_smoothed']
)
fig,ax = plt.subplots(2,1,figsize=(7.76, 6.48),constrained_layout=True,sharex=True)
ax[0].semilogx(T_RS.values,summary_df['m1_mean'],'r')
ax[0].semilogx(T_RS.values,smooth_summary_df['m1_smoothed'],'r--')
ax[0].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[0].set_ylabel('Proposed slope',size=16)
ax[0].set_xlim([0.01,10])
ax[1].semilogx(T_RS.values,summary_df['intercept_mean'],'k')
ax[1].semilogx(T_RS.values,smooth_summary_df['inter_smoothed'],'k--')
ax[1].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[1].set_ylabel('Proposed intercept',size=16)
ax[1].set_xlabel('Vibration Period (s)',size=16)
plt.savefig(os.path.join(out_dir,"Proposed slope smoothed.png"))
fig,ax = plt.subplots(2,1,figsize=(7.76, 6.48),constrained_layout=True,sharex=True)
ax[0].semilogx(T_RS.values,summary_df['chigh_mean'],'r')
ax[0].semilogx(T_RS.values,smooth_summary_df['chigh_smoothed'],'r--')
ax[0].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[0].set_ylabel('Proposed high offset',size=16)
ax[0].set_xlim([0.01,10])
ax[1].semilogx(T_RS.values,summary_df['clow_mean'],'k')
ax[1].semilogx(T_RS.values,smooth_summary_df['clow_smoothed'],'k--')
ax[1].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[1].set_ylabel('Proposed low offset',size=16)
ax[1].set_xlabel('Vibration Period (s)',size=16)
plt.savefig(os.path.join(out_dir,"Proposed offset smoothed.png"))
#%% Model for Paper 1
specific_T_values = [0.01,0.05, 0.1, 0.15,0.2, 0.3, 0.4,  0.5,0.75,1, 1.5,2, 3,4,5,7.5,10] 
filtered_indices = [np.argmin(np.abs(T_RS.values - T)) for T in specific_T_values]
filtered_T_RS = T_RS.values[filtered_indices]
filtered_m1_smoothed = smooth_summary_df['m1_smoothed'].values[filtered_indices]
filtered_m2_smoothed = smooth_summary_df['inter_smoothed'].values[filtered_indices]
fig, ax = plt.subplots(constrained_layout=True, sharex=True)
# fig.text(0.0, 1, '(b)', fontsize=16, fontweight='bold', ha='left', va='top')
ax.semilogx(filtered_T_RS, filtered_m1_smoothed, 'g', linewidth=2, linestyle='-', label='Slope')
ax.semilogx(filtered_T_RS, filtered_m2_smoothed, 'b', linewidth=2, linestyle='-', label='Intercept')
ax.legend(fontsize=20, loc='best')
ax.tick_params(direction='in', axis='both', which='both', labelsize=20)
ax.set_xlabel('Vibration Period (s)', size=20)
ax.set_xlim([0.01, 10])
ax.set_ylabel('Model parameters', size=20)
plt.savefig(os.path.join(out_dir, "BasinSlopeModelSmoothed.pdf"))
#%% Fbasin Vs. T for different T0
fig, ax = plt.subplots(figsize=(9.44, 7.39), constrained_layout=True)

SimIMs = pd.read_csv(data_dir1 / "im_sim.csv")
T0_test = [0.3,0.6,0.75,0.95,1.5,3]
colors = ['blue', 'red', 'orange','green','red','blue']
linestyles = ['--', '--', '--','-', '-','-.']
periods = [0.01,0.05, 0.1, 0.15,0.2, 0.3, 0.4,  0.5, 0.75, 0.869749002617783,1,1.5,2, 3,4,5,7.5,10] 
specific_T_values = ['pSA_0.010000000000', 'pSA_0.05', 'pSA_0.1', 'pSA_0.151991108295', 'pSA_0.2', 'pSA_0.3','pSA_0.4', 'pSA_0.5', 'pSA_0.75', 'pSA_0.869749002618', 'pSA_1.0','pSA_1.519911082953', 'pSA_2.0', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 'pSA_7.5', 'pSA_10.0']
adjfactor_dict = {}
for i,T0 in enumerate(T0_test):
    adjfactors = []
    for col in specific_T_values:
        # col = f"pSA_{T:.12f}"
        m1            = smooth_summary_df.loc[col]['m1_smoothed']
        intercept     = smooth_summary_df.loc[col]['inter_smoothed']
        c_high        = smooth_summary_df.loc[col]['chigh_smoothed']
        c_low         = smooth_summary_df.loc[col]['clow_smoothed']
        Period        = float(col.split('_')[1])
        adjfactors.append(TiwariMFinalUn(Period,float(T0),m1,intercept,c_low,c_high,thresh1,thresh2,thresh3,thresh4,thresh11))
    
    ax.semilogx(periods, adjfactors,  color=colors[i], linestyle=linestyles[i],linewidth=2,label="$T_{0}$" f" = {T0}")

ax.legend(fontsize=20, loc='best')
ax.tick_params(direction='in', axis='both', which='both', labelsize=26)
ax.set_xlabel('Vibration Period (s)', size=26)
ax.set_xlim([0.01, 10])
ax.set_ylim([-0.5, 0.5])
ax.tick_params(axis='x', pad=10)
ax.set_ylabel('Adjustment factor, $f_{UnmodBasin}$', size=26)
plt.savefig(os.path.join(out_dir, "AdjustmentfactorBasin.pdf"))
#%% Plotting p-values
pSA_values = T_RS
p_value1 = summary_df['p_mean']
p_value2 = summary_df['p_intmean']
# x_labels = [pSA.replace('pSA_', '') for pSA in pSA_values]
plt.figure(constrained_layout=True)
# plt.text(0.0035, 0.72, '(a)', fontsize=16, fontweight='bold', ha='left', va='top')
plt.semilogx(T_RS, p_value1, color = 'g',linewidth=2,label='Slope')
plt.semilogx(T_RS, p_value2, color = 'b',linewidth=2,label='Intercept')
plt.axhline(0.05,color='r',linestyle='--')
plt.xlim([0.01,10])
plt.tick_params(direction='in', axis='both', which='both',labelsize=20)
plt.xlabel('Vibration Period (s)',size=20)
plt.ylabel('p-values',size=20)
plt.legend(fontsize=20)
plt.savefig(os.path.join(out_dir,"Basinp-valueslopes.pdf"))
#%% Applying the factors to original Sims
SimIMs      = pd.read_csv(data_dir1 / "im_sim.csv")
for index,row in SimIMs.iterrows():
    for col in SimIMs.columns[9:]:
        m1            = smooth_summary_df.loc[col]['m1_smoothed']
        intercept     = smooth_summary_df.loc[col]['inter_smoothed']
        c_high        = smooth_summary_df.loc[col]['chigh_smoothed']
        c_low         = smooth_summary_df.loc[col]['clow_smoothed']
        Period        = float(col.split('_')[1])
        adjfactor = []
        station=row['stat_id']
        ind = stations.index[stations['stat_id']==station][0]
        T     = stations.iloc[ind]['T0']
        stat_group = stations.iloc[ind]['Basin Type']
        geom = stations.iloc[ind]['Geomorphology']
        if (stat_group == "Unmodeled Basin"):
            adjfactor = TiwariMFinalUn(Period,T,m1,intercept,c_low,c_high,thresh1,thresh2,thresh3,thresh4,thresh11)
        else:
            adjfactor = 0
        SimIMs.at[index,col]*=np.exp(adjfactor)    
Simadj_fp = os.path.join("Model Inputs", "im_simMFinalBasinModelDraft2.csv")
SimIMs.to_csv(Simadj_fp, index=False)
#%% MER
import numpy as np
import pandas as pd
from pathlib import Path
from mera import utils
from mera.mera_pymer4 import run_mera
data_dir = Path(__file__).parent / "Model Inputs"
output_dir = Path(__file__).parent / "Model Outputs"
output_dir.mkdir(exist_ok=True)
stations_ffp = data_dir / "stations.csv"
events_ffp = data_dir / "events.csv"
obs_ffp = data_dir / "im_obs.csv"
sim_ffp = data_dir / "im_simMFinalBasinModelDraft2.csv"
stations_df = pd.read_csv(stations_ffp, index_col=0)
events_df = pd.read_csv(events_ffp, index_col=0)
obs_df = pd.read_csv(obs_ffp, index_col=0)
sim_df = pd.read_csv(sim_ffp, index_col=0)
assert (
    np.all(obs_df.index == sim_df.index)
    and np.all(obs_df.event_id == sim_df.event_id)
    and np.all(obs_df.stat_id == sim_df.stat_id)
)
ims = [
    cur_im
    for cur_im in np.intersect1d(obs_df.columns, sim_df.columns)
    if cur_im.startswith("pSA")
    or cur_im in ["PGA", "CAV", "Ds575", "Ds595", "AI", "PGV"]
]
res_df = np.log(obs_df[ims] / sim_df[ims])
res_df["event_id"] = obs_df.event_id
res_df["stat_id"] = obs_df.stat_id
res_df["event_id"] = np.char.add("event_", res_df["event_id"].values.astype(str))
res_df["stat_id"] = np.char.add("stat_", res_df["stat_id"].values.astype(str))
mask = None
compute_site_term = True
results = run_mera(
    res_df,
    list(ims),
    "event_id",
    "stat_id",
    compute_site_term=compute_site_term,
    mask=mask,
    verbose=True,
    raise_warnings=True,
    min_num_records_per_event=3,
    min_num_records_per_site=3,
)
results.event_res_df.to_csv(output_dir / "event_residualsMFinalBasinModelDraft2.csv", index_label="event_id")
results.rem_res_df.to_csv(output_dir / "remaining_residualsMFinalBasinModelDraft2.csv", index_label="gm_id")
results.bias_std_df.to_csv(output_dir / "bias_stdMFinalBasinModelDraft2.csv", index_label="IM")
results.site_res_df.to_csv(output_dir / "site_residualsMFinalBasinModelDraft2.csv", index_label="stat_id")
results.site_cond_std_df.to_csv(output_dir / "station_cond_stdMFinalBasinModelDraft2.csv", index_label="stat_id")
#%% Reduction in phiS2S - All sites
output_dir           = Path(os.path.join(os.getcwd(),"Model Outputs"))
varcomps_all_new     = pd.read_csv(Path(output_dir/"bias_stdMFinalBasinModelDraft2.csv"),index_col=0)
index_pSa            = varcomps_all[6:].index
ReductionphiS2S      = ((varcomps_all.loc[index_pSa,'phiS2S']- varcomps_all_new.loc[index_pSa,'phi_S2S'])/varcomps_all.loc[index_pSa,'phiS2S'])*100
varcomps_all['modbias']     = varcomps_all['bias'].abs()
varcomps_all_new['modbias'] = varcomps_all_new['bias'].abs()
fig,ax               = plt.subplots(figsize=(9.94,6.36),constrained_layout=True)
ax.semilogx(varcomps_all[6:]['imVal'].values,varcomps_all.loc[index_pSa,'bias'],'b',linewidth=2,label='Old bias')
ax.semilogx(varcomps_all[6:]['imVal'].values,varcomps_all_new.loc[index_pSa,'bias'],'r',linewidth=2,label='New bias')
ax.axhline(y=0,color='maroon',linestyle='--',linewidth=2)
ax.set_ylabel('Bias',size=20)
ax.set_xlabel('Vibration Period (s)',size=20)
ax.set_xlim([0.01,10])
ax.legend(fontsize=20)
ax.tick_params(direction='in', axis='both', which='both',labelsize=20)
plt.savefig(os.path.join(out_dir,'Old&Newbias.png'))
fig,ax               = plt.subplots(figsize=(9.94,6.36),constrained_layout=True)       
ax.semilogx(varcomps_all[6:]['imVal'].values,ReductionphiS2S,'r',linewidth=2,label='Reduction in phiS2S')
ax.axhline(y=0,color='maroon',linestyle='--',linewidth=2)
ax.tick_params(direction='in', axis='both', which='both',labelsize=20)
ax.set_ylabel('Reduction in $\phi_{S2S}$ (%)',size=20)
ax.set_xlabel('Vibration Period (s)',size=20)
ax.set_xlim([0.01,10])
plt.savefig(os.path.join(out_dir,'RedphiS2S.png'))
fig,ax               = plt.subplots(figsize=(9.94,6.36),constrained_layout=True)
ax.semilogx(varcomps_all[6:]['imVal'].values,np.array(varcomps_all_new.loc[index_pSa,'modbias']-varcomps_all.loc[index_pSa,'modbias']),'k',linewidth=2)
ax.axhline(y=0,color='maroon',linestyle='--',linewidth=2)
ax.set_ylabel('Change in Bias',size=20)
ax.set_xlabel('Vibration Period (s)',size=20)
ax.set_xlim([0.01,10])
ax.tick_params(direction='in', axis='both', which='both',labelsize=20)
plt.savefig(os.path.join(out_dir,'Changebias.png'))
#%% Computation of a+dS2S for unmodelled basin sites
nstationsmod    = stations.copy()
nstationsmod['stat_id'] = 'stat_'+ nstationsmod['stat_id'].astype(str)
olddS2SUn       = dS2Sres_all.loc[ostationsmod[ (ostationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
olddS2SerrUn    = dS2Serr_all.loc[ostationsmod[ (ostationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
olddS2SMo       = dS2Sres_all.loc[ostationsmod[ (ostationsmod['Basin Type'] == 'Type 1 Basin')|(ostationsmod['Basin Type'] == 'Type 3 Basin') | (ostationsmod['Basin Type'] == 'Type 4 Basin')]['stat_id']]
olddS2SerrMo    = dS2Serr_all.loc[ostationsmod[ (ostationsmod['Basin Type'] == 'Type 1 Basin')|(ostationsmod['Basin Type'] == 'Type 3 Basin') | (ostationsmod['Basin Type'] == 'Type 4 Basin')]['stat_id']]
aold            = varcomps_all.iloc[6:,varcomps_all.columns.get_loc('bias')]
anew            = varcomps_all_new.iloc[6:,varcomps_all_new.columns.get_loc('bias')]
newdS2Sres_all  = pd.read_csv(output_dir / "site_residualsMFinalBasinModelDraft2.csv",index_col=0)
newdS2Sres_all  = newdS2Sres_all.dropna(axis=1)
newdS2Sres_all  = newdS2Sres_all.iloc[:,6:]
sorted_columns  = sorted(newdS2Sres_all.columns, key=lambda x: float(x.split('_')[1]))
newdS2Sres_all  = newdS2Sres_all[sorted_columns]
anew            = anew[sorted_columns]
newdS2SUn       = newdS2Sres_all.loc[nstationsmod[ (nstationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
newdS2SMo       = newdS2Sres_all.loc[nstationsmod[ (nstationsmod['Basin Type'] == 'Type 1 Basin')|(nstationsmod['Basin Type'] == 'Type 3 Basin') | (nstationsmod['Basin Type'] == 'Type 4 Basin')]['stat_id']]
oldadS2SUn      = aold+olddS2SUn
oldadS2SMo      = aold+olddS2SMo
newadS2SUn      = anew+newdS2SUn
newadS2SMo      = anew+newdS2SMo
fig,ax          = plt.subplots(figsize=(8.52,6.19),constrained_layout=True)
for i in range(len(newadS2SUn)):
    ax.semilogx(T_RS.values,newadS2SUn.iloc[i,:],'gray',linewidth=2)
ax.semilogx(T_RS.values,oldadS2SUn.mean(),'r',label='Original mean')
ax.semilogx(T_RS.values,newadS2SUn.mean(),'b',label='After adjustments mean')
ax.semilogx(T_RS.values,newadS2SUn.mean()+newadS2SUn.std(),'b--')
ax.semilogx(T_RS.values,newadS2SUn.mean()-newadS2SUn.std(),'b--')
ax.axhline(0,color='maroon',linestyle='--')
ax.grid(True, color='gray',which='both', linestyle='--', linewidth=0.5)
ax.legend(fontsize=16)
ax.set_ylabel('Systematic residual $a\ +\ \delta S2S_{S}$ for unmodelled basin sites',size=16)
ax.set_xlabel('Vibration Period, T (s)',size=16)
ax.tick_params(labelsize=14,direction='in',axis='both',which='both')
ax.set_xlim([0.01,10])
plt.savefig(os.path.join(out_dir,'SysresidualUnmodeledsites.png'))
#%% Cat. Mean and cat standard deviation plots (Just dS2S plots)
newdS2Serr_all  = pd.read_csv(output_dir / "station_cond_stdMFinalBasinModelDraft2.csv",index_col=0)
newdS2Serr_all  = newdS2Serr_all.dropna(axis=1)
newdS2Serr_all  = newdS2Serr_all.iloc[:,6:]
sorted_columns  = sorted(newdS2Serr_all.columns, key=lambda x: float(x.split('_')[1]))
newdS2Serr_all  = newdS2Serr_all[sorted_columns]
newdS2SerrUn    = newdS2Serr_all.loc[nstationsmod[ (nstationsmod['Basin Type'] == 'Unmodeled Basin')]['stat_id']]
newdS2SerrMo    = newdS2Serr_all.loc[nstationsmod[ (nstationsmod['Basin Type'] == 'Type 1 Basin')|(nstationsmod['Basin Type'] == 'Type 3 Basin') | (nstationsmod['Basin Type'] == 'Type 4 Basin')]['stat_id']]
oldstdall       = np.sqrt(np.mean(dS2Sres_all**2,axis=0)+np.mean(dS2Serr_all**2,axis=0))
newstdall       = np.sqrt(np.mean(newdS2Sres_all**2,axis=0)+np.mean(newdS2Serr_all**2,axis=0))
oldUnstd      = np.sqrt(np.mean(olddS2SUn**2,axis=0)+np.mean(olddS2SerrUn**2,axis=0))
oldMostd      = np.sqrt(np.mean(olddS2SMo**2,axis=0)+np.mean(olddS2SerrMo**2,axis=0))
newUnstd      = np.sqrt(np.mean(newdS2SUn**2,axis=0)+np.mean(newdS2SerrUn**2,axis=0))
newMostd      = np.sqrt(np.mean(newdS2SMo**2,axis=0)+np.mean(newdS2SerrMo**2,axis=0))
fig,axes    = plt.subplots(2,1,figsize = (8.52, 7.62),constrained_layout=True,sharex=True)
axes[0].semilogx(T_RS.values,olddS2SUn.mean(),'g',linewidth=4,label='Unmodeled sites')
axes[0].semilogx(T_RS.values,newdS2SUn.mean(),'g--',linewidth=4,label='Unmodeled sites (Proposed model)')
axes[0].axhline(0,color='k',linestyle='--',linewidth=4)
axes[0].axhline(0,color='k',linestyle='-',linewidth=2)
axes[1].semilogx(T_RS.values,oldstdall,'k',linewidth=4,label='All sites')
axes[1].semilogx(T_RS.values,oldUnstd,'g',linewidth=4,label='Unmodeled sites')
axes[1].semilogx(T_RS.values,newstdall,'k--',linewidth=4,label='All sites (Proposed model)')
axes[1].semilogx(T_RS.values,newUnstd,'g--',linewidth=4,label='Unmodeled sites (Proposed model)')
axes[0].text(0.0105,0.07,'Underprediction',size=16,fontweight='bold')
axes[0].text(0.0105,-0.07,'Overprediction',size=16,fontweight='bold')
axes[0].set_ylabel('$\delta S2S_{S}$',size=16)
axes[0].tick_params(labelsize=14,direction='in', axis='both', which='both')
axes[0].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)   
axes[1].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes[1].set_ylabel('Standard deviation, $\phi_{S2S}$',size=16)
axes[1].set_xlabel('Vibration Period, $\it T\ (s)$', size=16)
axes[1].legend(fontsize=16,frameon=False)
axes[1].tick_params(labelsize=14,direction='in', axis='both', which='both')
axes[1].set_xlim([0.01, 10])
axes[1].set_ylim([0,None])
plt.savefig(os.path.join(out_dir,'deltaS2SUnmodeledsites.png'))
#%% Cat. Mean and cat standard deviation plots
plt.rcParams["font.family"] = "Times New Roman"
Allsitesres     = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\9. Model development\Results_combined.csv"),index_col=0)
fig,axes    = plt.subplots(2,1,figsize = (11.21,  9.54),constrained_layout=True,sharex=True)
fig.text(0.001, 1, '(a)', fontsize=36, fontweight='bold', ha='left', va='top')
axes[0].semilogx(T_RS.values,oldadS2SMo.mean(),'#19e6b2',linewidth=2,label='Modeled basin sites')
axes[0].semilogx(T_RS.values,oldadS2SUn.mean(),'#e6194B',linewidth=2,label='Unmodeled basin sites')
axes[0].semilogx(T_RS.values,newadS2SUn.mean(),'#e6194B',linestyle='--',linewidth=2,label='Unmodeled basin sites (Proposed model)')
# axes[0].semilogx(T_RS.values,newadS2SMo.mean(),'g',linestyle='--',linewidth=2,label='Modeled basin sites (Proposed model)')
axes[0].axhline(0,color='k',linestyle='--',linewidth=1)
axes[0].semilogx(T_RS.values,aold+dS2Sres_all.mean(),'k',linewidth=2)
axes[0].semilogx(T_RS.values,Allsitesres['anew'].values+newdS2Sres_all.mean(),'k--',linewidth=2)
axes[1].semilogx(T_RS.values,oldMostd,'#19e6b2',linewidth=2,label='Modeled basin sites')
axes[1].semilogx(T_RS.values,oldUnstd,'#e6194B',linewidth=2,label='Unmodeled basin sites (Original)')
axes[1].semilogx(T_RS.values,oldstdall,'k',linewidth=2,label='All sites (Original)')
axes[1].semilogx(T_RS.values,newUnstd,'#e6194B',linestyle = '--',linewidth=2,label='Unmodeled basin sites (Proposed model)')
axes[1].semilogx(T_RS.values,Allsitesres['newstd'].values,'k--',linewidth=2,label='All sites (All parametric models)')
axes[0].text(0.0105,0.25,'Underprediction',size=30,fontstyle='italic')
axes[0].text(0.0105,-0.27,'Overprediction',size=30,fontstyle='italic')
axes[0].set_ylabel('$a\ +\ \delta S2S_{S}$',size=36)
axes[0].tick_params(labelsize=29,direction='in', axis='both', which='both')
axes[0].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)   
axes[1].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
axes[1].set_ylabel('$\phi_{S2S}$',size=36)
axes[1].set_xlabel('Vibration Period, T (s)', size=36)
axes[1].legend(fontsize=22,frameon=False)
axes[1].tick_params(labelsize=32,direction='in', axis='both', which='both')
axes[1].set_xlim([0.01, 10])
axes[1].set_ylim([0,None])
axes[1].tick_params(axis='x', pad=10)
axes[0].set_ylim([-0.3,0.3])
plt.savefig(os.path.join(out_dir,'Sysresidual&stdUnsites.pdf'))
#%% % Reduction in phiS2S
fig,ax = plt.subplots(figsize=(7.43, 6.0),constrained_layout=True)
fig.text(0.001, 1, '(b)', fontsize=22, fontweight='bold', ha='left', va='top')
Red = ((oldUnstd-newUnstd)/oldUnstd)*100
ax.semilogx(T_RS.values,Red,'#e6194B',linewidth=2,label='Unmodeled Basin sites')
ax.set_xlabel('Vibration Period, T (s)', size=22)
ax.semilogx(T_RS.values,Allsitesres['Reduction'],'k',linewidth=2, label = 'All sites')
ax.set_ylabel('Reduction in $\phi_{S2S}$ (%)',size=22)
ax.tick_params(labelsize=20,direction='in', axis='both', which='both')
ax.set_xlim([0.01, 10])
ax.legend(fontsize=20,frameon=False)
ax.grid(True, color='gray',which='both', linestyle='--', linewidth=0.4,alpha=0.5)
ax.set_ylim([0,None])
ax.tick_params(axis='x', pad=5)
plt.savefig(os.path.join(out_dir,'RedinphiS2SUnmod.pdf'))
#%% Lowess check before and after
# IMs        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4']  
# IMs        = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
IMs        = ['pSA_0.869749002618','pSA_2.0','pSA_3.0','pSA_5.0'] 
# IMs          = ['pSA_6.135907273413','pSA_7.5','pSA_8.697490026178','pSA_10.0']
mask         = T0basinUn.notna()
masknew      = mask.copy()
x            = T0basinUn[mask]
x            = np.array(x)
y            = dS2SbasinUn
z            = newdS2SUn
order        = np.argsort(x)
fig,ax  = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2)
    dS2S_sel = y.loc[:,im]
    mask.index = dS2S_sel.index
    dS2S_sel = dS2S_sel[mask]
    dS2S_selnew = z.loc[:,im]    
    masknew.index = dS2S_selnew.index    
    dS2S_selnew = dS2S_selnew[masknew]
    ax[i,j].scatter(x, np.array(dS2S_selnew), color='#008080',edgecolor='k')
    lowessold,lowessoldstd = loess(x,np.array(dS2S_sel),0.3)
    lowessnew,lowessoldnew = loess(x,np.array(dS2S_selnew),0.3)
    ax[i,j].plot(x[order], lowessold[order], color='#FF6F61', linewidth=2,label='Before adjustment')
    ax[i,j].fill_between(x[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='#FF6F61', alpha=0.1)
    ax[i,j].plot(x[order], lowessnew[order], color='#008080',linewidth=2, label='After adjustment')
    ax[i,j].fill_between(x[order], lowessnew[order] - 1.96*lowessnew[order], lowessnew[order] + 1.96*lowessnew[order], color='#008080', alpha=0.1)
    ax[i, j].axhline(0, color='k', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    ax[i, j].set_xscale('log')
global_y_min, global_y_max = float('inf'), float('-inf')
for i in range(2):  # rows and cols define the subplot grid
    for j in range(2):
         y_min, y_max = ax[i, j].get_ylim()
         global_y_min = min(global_y_min, y_min)
         global_y_max = max(global_y_max, y_max)

y0= 0
max_y_range = max(abs(global_y_max - y0), abs(global_y_min - y0))
rounded_value = np.round(float(im.split('pSA_')[1]), 2)
for i in range(2):
    for j in range(2):
        ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range])
ax[0][0].legend(fontsize=16)
ax[1][0].set_xlabel('$T_0$ (s)', size=20)
ax[1][1].set_xlabel('$T_0$ (s)', size=20)
# plt.savefig(os.path.join(out_dir,"Loessbefore_afterveryshort.png"))
# plt.savefig(os.path.join(out_dir,"Loessbefore_aftershort.png"))
plt.savefig(os.path.join(out_dir,"Loessbefore_afterlong.pdf"))
# plt.savefig(os.path.join(out_dir,"Loessbefore_afterverylong.png"))

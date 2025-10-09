# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:43:49 2024

@author: ati47
"""

import numpy as np
import math
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import addcopyfighandler
# import statsmodels.api as sm
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import os
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
from scipy.stats import linregress
from scipy.stats import t
#%% Load all data
data_dir1           = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\New Results")
ObsIMs              = pd.read_csv(Path(data_dir1 / "im_obs.csv"))
SimIMs              = pd.read_csv(data_dir1 / "im_sim.csv")
dS2Sres_all         = pd.read_csv(data_dir1/ "Residuals"/"PJSreStationBiased_sim.csv",index_col=0)
dS2Serr_all         = pd.read_csv(data_dir1/"Residuals"/"PJSreStationerror_sim.csv",index_col=0)
varcomps_all        = pd.read_csv(data_dir1/"Residuals"/"PJSvarCompsBiased_sim.csv",index_col=0)
Features            = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\9. Model development\Correlation\Features.csv"),index_col = 'Station Code')
Features.index      = 'Station_' + Features.index.astype(str)
stations            = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\New Results\stations.csv"))
stations            = pd.merge(stations[['stat_id','stat_name']],Features[['stat_name','Geomorphology','H1250','H1500']],on='stat_name')
phiS2S_RS           = varcomps_all.iloc[6:,3]
T_RS                = varcomps_all.iloc[6:,0]
dS2Sres_all         = dS2Sres_all.dropna(axis=1)
dS2Sres_all         = dS2Sres_all.transpose()
dS2Serr_all         = dS2Serr_all.dropna(axis=1)
dS2Serr_all         = dS2Serr_all.transpose()
H1500               = stations['H1500']
H1250hill           = stations[stations['Geomorphology']=="Hill"]["H1250"]
ostationsmod        = stations.copy()
nstationsmod        = stations.copy()
ostationsmod['stat_id'] = 'Station_'+ ostationsmod['stat_id'].astype(str)
nstationsmod['stat_id'] = 'stat_'+ nstationsmod['stat_id'].astype(str)
dS2Sres_all         = dS2Sres_all.iloc[:,6:]
dS2Serr_all         = dS2Serr_all.iloc[:,6:]
olddS2Shill         = dS2Sres_all.loc[ostationsmod[ostationsmod['Geomorphology']=="Hill"]['stat_id']]
olddS2Serrhill      = dS2Serr_all.loc[ostationsmod[ostationsmod['Geomorphology']=="Hill"]['stat_id']]
varcomps            = varcomps_all.copy()
varcomps.index      = varcomps_all.index
Rai1500correction   = pd.read_csv("Rai1500correction.csv")
c_low_interpR1500   = np.interp(np.log10(T_RS.values),np.array(np.log10(Rai1500correction['imVal'])),np.array(Rai1500correction['clow']))
c_high_interpR1500  = np.interp(np.log10(T_RS.values),np.array(np.log10(Rai1500correction['imVal'])),np.array(Rai1500correction['chigh']))
out_dir             = Path(os.path.join(os.getcwd(),"Model Diagnostics","ModelsHillSites","M10FinalSlopeModelv2"))
#%% Slope Model M10
def TiwariM10(H,c_low,m1,m2,c_high,thresh1,thresh2):
    return np.piecewise(
        H,
        [H < -thresh1,
         (-thresh1<=H) & (H<=0),
         (0<=H) & (H<=thresh2),
         H>thresh2,
         ],
        [lambda H: c_low, 
         lambda H: m1*H,
         lambda H:  m2*H,
         lambda H: c_high]
    )
#%% Fixing thresholds
thresh1 = 50
thresh2 = 70
H1250model = np.append(H1250hill, [-thresh1, thresh2])
IMs = ['pSA_0.5', 'pSA_0.75', 'pSA_0.869749002618', 'pSA_1.0']
# IMs   = ['pSA_2.0','pSA_5.0','pSA_7.5','pSA_10.0']
fig, ax = plt.subplots(2, 2, figsize=(12.06, 8.05), constrained_layout=True, sharex=True, sharey=True)
fig2, ax2 = plt.subplots(2, 2, figsize=(12.06, 8.05), constrained_layout=True, sharex=True, sharey=True)
p_values_dict = {}
for idx, im in enumerate(IMs):
    mask1 = (H1250hill >= -thresh1) & (H1250hill <=0)
    mask2 = (H1250hill >= 0) & (H1250hill <=thresh2)
    olddS2ShillT = olddS2Shill[im]
    olddS2ShillT.index = H1250hill.index
    
    filtered_H1250hill1 = H1250hill[mask1]
    filtered_olddS2Shill1 = olddS2ShillT[mask1]
    filtered_H1250hill2 = H1250hill[mask2]
    filtered_olddS2Shill2 = olddS2ShillT[mask2]
    
    # Calculate m1
    m1 = np.sum(filtered_H1250hill1 * filtered_olddS2Shill1) / np.sum(filtered_H1250hill1**2)
    m2 = np.sum(filtered_H1250hill2 * filtered_olddS2Shill2) / np.sum(filtered_H1250hill2**2)
    
 
    # Calculate c_low and c_high based on m1
    c_low  = m1* -thresh1
    c_high = m2 * thresh2
    
    H1250hillval = np.asarray(H1250hill)
    olddS2ShillTval = np.asarray(olddS2ShillT)
    
    # Use the new c_low and c_high in TiwariM5
    ans = TiwariM10(H1250model,c_low, m1, m2,c_high,thresh1,thresh2)
    ans1 = TiwariM10(H1250hillval,c_low, m1, m2, c_high,thresh1,thresh2)
    
    ans2 = m1*filtered_H1250hill1
    ans3 = m2*filtered_H1250hill2
    
    ss_residual1 = np.sum((filtered_olddS2Shill1 - ans2)**2)
    ss_residual2 = np.sum((filtered_olddS2Shill2 - ans3)**2)
    n1 = len(filtered_H1250hill1)
    n2 = len(filtered_H1250hill2)
    t_stat1 = m1 / (np.sqrt(ss_residual1 / (n1 - 1)) / np.sqrt(np.sum(filtered_H1250hill1**2)))
    t_stat2 = m2 / (np.sqrt(ss_residual2 / (n2 - 1)) / np.sqrt(np.sum(filtered_H1250hill2**2)))
    from scipy.stats import t
    p_value1 = 2 * (1 - t.cdf(np.abs(t_stat1), df=n1-1))
    p_value2 = 2 * (1 - t.cdf(np.abs(t_stat2), df=n2-1))
    
    p_values_dict[im] = {'p_value1': p_value1, 'p_value2': p_value2}
    res = olddS2ShillTval - ans1
    slope, intercept, r_value, p_value, std_err = linregress(H1250hill, res)
    trendline = slope * H1250hill + intercept
    df = pd.DataFrame({'H1250hill': H1250model, 'dS2Shill': ans})
    df = df.sort_values(by='H1250hill')
    
    i, j = divmod(idx, 2)
    ax[i, j].scatter(H1250hill, olddS2Shill.loc[:, im], color='lightblue', edgecolor='k', s=40)
    ax[i, j].plot(df['H1250hill'], df['dS2Shill'], 'k')
    ax[i, j].axhline(0, color='maroon', linestyle='--')
    x0, y0= 0,0
    x_min, x_max = ax[i,j].get_xlim()
    y_min, y_max = ax[i,j].get_ylim()
    max_x_range = max(abs(x_max - x0), abs(x_min - x0))
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    ax[i, j].set_xlim([x0 - max_x_range, x0 + max_x_range])
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range])
    ax2[i, j].scatter(H1250hill, res, color='lightblue', edgecolor='k', s=40)
    ax2[i, j].plot(H1250hill, trendline, color='green', linestyle='--', label=f'Trendline (slope = {slope:.4f}, p = {p_value:.2f})', linewidth=1)
    
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    ax2[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax2[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    ax[i, j].axvline(-thresh1, color='k', linestyle='--')
    ax[i, j].axvline(thresh2, color='k', linestyle='--')
    ax2[i, j].legend()

ax[1][0].set_xlabel('$H_{1250}$ (m)', size=20)
ax[1][1].set_xlabel('$H_{1250}$ (m)', size=20)
fig.savefig(os.path.join(out_dir,"SampleModel.png"))
ax2[1][0].set_xlabel('$H_{1250}$ (m)', size=20)
ax2[1][1].set_xlabel('$H_{1250}$ (m)', size=20)
plt.savefig(os.path.join(out_dir,"SampleModelResiduals.png"))
#%% Bootstrapping slope - first distributions
n_bootstrap = 1000
H1250model = np.append(H1250hill, [-thresh1, thresh2])
IM       = ['pSA_0.3','pSA_0.75','pSA_1.0','pSA_3.0']
fig, ax = plt.subplots(2, 2, figsize=(12.06, 8.05), constrained_layout=True, sharex=True, sharey=True)
fig.text(0.0, 1, '(a)', fontsize=22, fontweight='bold', ha='left', va='top')
for idx, im in enumerate(IM):
    SitedS2SDB = pd.DataFrame({'H1250hill': H1250hill.values, 'dS2S': olddS2Shill[im].values})
    m1_dist = []
    m2_dist = []
    c_low_dist = []
    c_high_dist = []
    i, j = divmod(idx, 2)
    for _ in range(n_bootstrap):
        SitedS2SDB_bootstrap = SitedS2SDB.sample(frac=1, replace=True)
        mask1 = (SitedS2SDB_bootstrap['H1250hill'].values >= -thresh1) & (SitedS2SDB_bootstrap['H1250hill'].values <=0)
        mask2 = (SitedS2SDB_bootstrap['H1250hill'].values >= 0) & (SitedS2SDB_bootstrap['H1250hill'].values <=thresh2) 
        filtered_H1250hill1   = SitedS2SDB_bootstrap['H1250hill'][mask1]
        filtered_olddS2Shill1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
        filtered_H1250hill2   = SitedS2SDB_bootstrap['H1250hill'][mask2]
        filtered_olddS2Shill2 =  SitedS2SDB_bootstrap['dS2S'][mask2]

        # Calculate m1
        m1 = np.sum(filtered_H1250hill1 * filtered_olddS2Shill1) / np.sum(filtered_H1250hill1**2)
        m2 = np.sum(filtered_H1250hill2 * filtered_olddS2Shill2) / np.sum(filtered_H1250hill2**2)   
        m1_dist.append(m1)
        m2_dist.append(m2)
        c_low  = m1* -thresh1
        c_high = m2 * thresh2
        c_low_dist.append(c_low)
        c_high_dist.append(c_high)
        
        ans = TiwariM10(H1250model, c_low, m1,m2,c_high,thresh1,thresh2)
        df = pd.DataFrame({'H1250hill': H1250model, 'dS2Shill': ans})
        df = df.sort_values(by='H1250hill')
        ax[i, j].plot(df['H1250hill'],df['dS2Shill'],'gray',alpha=0.05)
    ax[i, j].scatter(H1250hill, olddS2Shill.loc[:, im], color='#993333', edgecolor='k', s=40)
    HM7_dist   = pd.DataFrame({'clow':c_low_dist,'chigh':c_high_dist,'m1':m1_dist,'m2':m2_dist})
    m1_mean    = HM7_dist['m1'].mean() 
    m2_mean    = HM7_dist['m2'].mean()
    clow_mean  = HM7_dist['clow'].mean()
    chigh_mean = HM7_dist['chigh'].mean()
    m1_std     = HM7_dist['m1'].std()
    m2_std     = HM7_dist['m2'].std()
    clow_std   = HM7_dist['clow'].std()
    chigh_std  = HM7_dist['chigh'].std()
    ans = TiwariM10(H1250model, clow_mean, m1_mean, m2_mean, chigh_mean,thresh1,thresh2)
    df = pd.DataFrame({'H1250hill': H1250model, 'dS2Shill': ans})
    df = df.sort_values(by='H1250hill')
    ax[i, j].plot(df['H1250hill'], df['dS2Shill'], 'k',linewidth=2)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i, j].axvline(0, color='k', linestyle='--')
    x0, y0= 0,0
    x_min, x_max = ax[i,j].get_xlim()
    y_min, y_max = ax[i,j].get_ylim()
    max_x_range = max(abs(x_max - x0), abs(x_min - x0))
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    ax[i, j].axvline(-thresh1, color='maroon', linestyle='--')
    ax[i, j].axvline(thresh2, color='maroon', linestyle='--')
    ax[i, j].set_xlim([x0 - max_x_range, x0 + max_x_range])
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range])
ax[1][0].set_xlabel('$H_{1250}$ (m)', size=20)
ax[1][1].set_xlabel('$H_{1250}$ (m)', size=20)
plt.savefig(os.path.join(out_dir,"SampleModelBootstrapping.pdf"))
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
#%% Sample Bootstrapping
n_bootstrap = 1000
H1250hill           = stations[stations['Geomorphology']=="Hill"]["H1250"]
order = np.argsort(np.array(H1250hill))

# IM        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4'] 
IM       = ['pSA_0.5','pSA_0.75','pSA_0.869749002618','pSA_1.0']
# IM       = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
# IM         = ['pSA_6.135907273413','pSA_7.5','pSA_8.697490026178','pSA_10.0']
fig, ax = plt.subplots(2, 2, figsize=(12.06, 8.05), constrained_layout=True, sharex=True, sharey=True)
for idx, im in enumerate(IM):
    i, j = divmod(idx, 2)
    ax[i, j].scatter(H1250hill, olddS2Shill.loc[:, im], color='#993333', edgecolor='k', s=40)
    lowessold,lowessoldstd = loess(np.array(H1250hill),np.array(olddS2Shill.loc[:,im]),0.3)
    ax[i,j].plot(np.array(H1250hill)[order], lowessold[order], color='k', linewidth=1)
    ax[i,j].fill_between(np.array(H1250hill)[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='k', alpha=0.1)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i, j].axvline(0, color='k', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    x_min, x_max = ax[i,j].get_xlim()
    y_min, y_max = ax[i,j].get_ylim()
    max_x_range = max(abs(x_max - x0), abs(x_min - x0))
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    ax[i, j].set_xlim([x0 - max_x_range, x0 + max_x_range])
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range])

ax[1][0].set_xlabel('$H_{1250}$ (m)', size=20)
ax[1][1].set_xlabel('$H_{1250}$ (m)', size=20)

# fig.savefig(os.path.join(out_dir,"H1250TrendShort.pdf"), dpi=300)
#%% Bootstrapping slope and loess
n_bootstrap = 1000
H1250hill           = stations[stations['Geomorphology']=="Hill"]["H1250"]
order = np.argsort(np.array(H1250hill))
H1250model = np.append(H1250hill, [-thresh1, thresh2])
# IM        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4'] 
IM       = ['pSA_0.3','pSA_0.75','pSA_1.0','pSA_3.0']
# IM       = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
# IM         = ['pSA_6.135907273413','pSA_7.5','pSA_8.697490026178','pSA_10.0']
fig, ax = plt.subplots(2, 2, figsize=(12.06, 8.05), constrained_layout=True, sharex=True, sharey=True)
for idx, im in enumerate(IM):
    SitedS2SDB = pd.DataFrame({'H1250hill': H1250hill.values, 'dS2S': olddS2Shill[im].values})
    m1_dist = []
    m2_dist = []
    c_low_dist = []
    c_high_dist = []
    i, j = divmod(idx, 2)
    bootstrap_curves = []
    for _ in range(n_bootstrap):
        SitedS2SDB_bootstrap = SitedS2SDB.sample(frac=1, replace=True)
        mask1 = (SitedS2SDB_bootstrap['H1250hill'].values >= -thresh1) & (SitedS2SDB_bootstrap['H1250hill'].values <=0)
        mask2 = (SitedS2SDB_bootstrap['H1250hill'].values >= 0) & (SitedS2SDB_bootstrap['H1250hill'].values <=thresh2) 
        filtered_H1250hill1   = SitedS2SDB_bootstrap['H1250hill'][mask1]
        filtered_olddS2Shill1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
        filtered_H1250hill2   = SitedS2SDB_bootstrap['H1250hill'][mask2]
        filtered_olddS2Shill2 =  SitedS2SDB_bootstrap['dS2S'][mask2]
        # Calculate m1
        m1 = np.sum(filtered_H1250hill1 * filtered_olddS2Shill1) / np.sum(filtered_H1250hill1**2)
        m2 = np.sum(filtered_H1250hill2 * filtered_olddS2Shill2) / np.sum(filtered_H1250hill2**2)   
        m1_dist.append(m1)
        m2_dist.append(m2)
        c_low  = m1* -thresh1
        c_high = m2 * thresh2
        c_low_dist.append(c_low)
        c_high_dist.append(c_high)       
        ans = TiwariM10(H1250model, c_low, m1,m2,c_high,thresh1,thresh2)
        bootstrap_curves.append(ans)
        df = pd.DataFrame({'H1250hill': H1250model, 'dS2Shill': ans})
        df = df.sort_values(by='H1250hill')
        # ax[i, j].plot(df['H1250hill'],df['dS2Shill'],'b',alpha=0.01)
        # Append the bootstrapped curve (new; don't plot it individually)
        bootstrap_curves.append(ans)
    
    # Convert to array for easy computation (new; shape: (n_bootstrap, len(H1250model)))
    bootstrap_curves = np.array(bootstrap_curves)
    
    # Compute pointwise mean and 95% CI from bootstraps (new; non-parametric)
    mean_ans = np.mean(bootstrap_curves, axis=0)
    lower_ci = np.percentile(bootstrap_curves, 2.5, axis=0)
    upper_ci = np.percentile(bootstrap_curves, 97.5, axis=0)

    ax[i, j].scatter(H1250hill, olddS2Shill.loc[:, im], color='#993333', edgecolor='k', s=40)
    lowessold,lowessoldstd = loess(np.array(H1250hill),np.array(olddS2Shill.loc[:,im]),0.3)
    ax[i,j].plot(np.array(H1250hill)[order], lowessold[order], color='k', linewidth=2, label='LOWESS')
    ax[i,j].fill_between(np.array(H1250hill)[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='k', alpha=0.1)
    HM7_dist   = pd.DataFrame({'clow':c_low_dist,'chigh':c_high_dist,'m1':m1_dist,'m2':m2_dist})
    m1_mean    = HM7_dist['m1'].mean() 
    m2_mean    = HM7_dist['m2'].mean()
    clow_mean  = HM7_dist['clow'].mean()
    chigh_mean = HM7_dist['chigh'].mean()
    m1_std     = HM7_dist['m1'].std()
    m2_std     = HM7_dist['m2'].std()
    clow_std   = HM7_dist['clow'].std()
    chigh_std  = HM7_dist['chigh'].std()
    ans = TiwariM10(H1250model, clow_mean, m1_mean, m2_mean, chigh_mean,thresh1,thresh2)
    df = pd.DataFrame({'H1250hill': H1250model, 'dS2Shill': ans})
    df = df.sort_values(by='H1250hill')
    ax[i, j].plot(df['H1250hill'], df['dS2Shill'], 'b',linewidth=2,label='Model')
    ax[i, j].fill_between(df['H1250hill'], lower_ci[df.index], upper_ci[df.index], color='b', alpha=0.1)
    ax[i, j].axhline(0, color='k', linestyle='--')
    ax[i, j].axvline(0, color='k', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    x0, y0= 0,0
    x_min, x_max = ax[i,j].get_xlim()
    y_min, y_max = ax[i,j].get_ylim()
    max_x_range = max(abs(x_max - x0), abs(x_min - x0))
    max_y_range = max(abs(y_max - y0), abs(y_min - y0))
    ax[i, j].axvline(-thresh1, color='maroon', linestyle='--')
    ax[i, j].axvline(thresh2, color='maroon', linestyle='--')
    ax[i, j].set_xlim([x0 - max_x_range, x0 + max_x_range])
    ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range])
ax[0][0].legend(fontsize=14)
ax[1][0].set_xlabel('$H_{1250}$ (m)', size=20)
ax[1][1].set_xlabel('$H_{1250}$ (m)', size=20)
# plt.savefig(os.path.join(out_dir,"SampleModelBootstrappingVeryShort.png"))
plt.savefig(os.path.join(out_dir,"SampleModelBootstrappingShort.pdf"))
# plt.savefig(os.path.join(out_dir,"SampleModelBootstrappingLong.png"))
# plt.savefig(os.path.join(out_dir,"SampleModelBootstrappingVeryLong.png"))
#%% Histograms for m1 and m2 for period of 1s
m1 = HM7_dist['m1']
m2 = HM7_dist['m2']

# Calculate means
mean_m1 = np.mean(m1)
mean_m2 = np.mean(m2)

# Create the 1x2 subplot layout with shared axes
fig, axes = plt.subplots(2, 1, figsize=(6.94, 7.34),constrained_layout = True, sharey=True,sharex=True)
fig.text(0.0, 1, '(b)', fontsize=26, fontweight='bold', ha='left', va='top')
# Plot the histogram for m1
axes[0].hist(m1, bins=20, color='g', alpha=0.7, edgecolor='black')
axes[0].axvline(mean_m1, color='red', linestyle='--', linewidth=2.5)
axes[0].set_xlabel('$h_1$ (Lower slope)',size=26)
axes[0].set_ylabel('Frequency',size=26)
axes[0].tick_params(direction='in', axis='both', which='both', labelsize=26)

# Plot the histogram for m2
axes[1].hist(m2, bins=20, color='b', alpha=0.7, edgecolor='black')
axes[1].axvline(mean_m2, color='red', linestyle='--', linewidth=2.5)
axes[1].set_xlabel('$h_2$ (Upper slope)',size=26)
axes[1].set_ylabel('Frequency',size=26)
axes[1].tick_params(direction='in', axis='both', which='both', labelsize=26)
plt.savefig(os.path.join(out_dir,"SlopeHistograms.pdf"))
#%% Do bootstrap for all periods
summary_stats = []
for im in dS2Sres_all.columns:
    SitedS2SDB = pd.DataFrame({'H1250hill': H1250hill.values, 'dS2S': olddS2Shill[im].values})
    m1_dist = []
    m2_dist = []
    c_low_dist = []
    c_high_dist = []
    for _ in range(n_bootstrap):
        SitedS2SDB_bootstrap = SitedS2SDB.sample(frac=1, replace=True)
        mask1 = (SitedS2SDB_bootstrap['H1250hill'].values >= -thresh1) & (SitedS2SDB_bootstrap['H1250hill'].values <=0)
        mask2 = (SitedS2SDB_bootstrap['H1250hill'].values >= 0) & (SitedS2SDB_bootstrap['H1250hill'].values <=thresh2) 
        filtered_H1250hill1   = SitedS2SDB_bootstrap['H1250hill'][mask1]
        filtered_olddS2Shill1 =  SitedS2SDB_bootstrap['dS2S'][mask1]
        filtered_H1250hill2   = SitedS2SDB_bootstrap['H1250hill'][mask2]
        filtered_olddS2Shill2 =  SitedS2SDB_bootstrap['dS2S'][mask2]   
        m1 = np.sum(filtered_H1250hill1 * filtered_olddS2Shill1) / np.sum(filtered_H1250hill1**2)
        m2 = np.sum(filtered_H1250hill2 * filtered_olddS2Shill2) / np.sum(filtered_H1250hill2**2)
        c_low  = m1* -thresh1
        c_high = m2 * thresh2
        m1_dist.append(m1)
        m2_dist.append(m2)
        c_low_dist.append(c_low)
        c_high_dist.append(c_high)        
        HM7_dist   = pd.DataFrame({'clow':c_low_dist,'chigh':c_high_dist,'m1':m1_dist,'m2':m2_dist})
        m1_mean    = HM7_dist['m1'].mean() 
        m2_mean    = HM7_dist['m2'].mean()
        clow_mean  = HM7_dist['clow'].mean()
        chigh_mean = HM7_dist['chigh'].mean()
        m1_std     = HM7_dist['m1'].std()
        m2_std     = HM7_dist['m2'].std()
        clow_std   = HM7_dist['clow'].std()
        chigh_std  = HM7_dist['chigh'].std()
    summary_stats.append({
        'IM_sel': im,
        'm1_mean': m1_mean,
        'm2_mean': m2_mean,
        'clow_mean': clow_mean,
        'chigh_mean': chigh_mean,
        'm1_std':m1_std,
        'm2_std':m2_std,
        'clow_std': clow_std,
        'chigh_std': chigh_std
    })
summary_df = pd.DataFrame(summary_stats)
summary_df.set_index('IM_sel', inplace=True)
# summary_df.to_excel(os.path.join(out_dir,"Bootstrapping_results.xlsx"))
#%% Finding p-value corresponding to an IM - determining right slope
summary_df = pd.read_excel(os.path.join(out_dir,"Bootstrapping_results.xlsx"),index_col=0)
thresh1 = 50
thresh2 = 70
H1250model = np.append(H1250hill, [-thresh1, thresh2])
p_values_dict = {}
for im in dS2Sres_all.columns:
    mask1 = (H1250hill >= -thresh1) & (H1250hill <=0)
    mask2 = (H1250hill >= 0) & (H1250hill <=thresh2)
    olddS2ShillT = olddS2Shill[im]
    olddS2ShillT.index = H1250hill.index
    
    filtered_H1250hill1 = H1250hill[mask1]
    filtered_olddS2Shill1 = olddS2ShillT[mask1]
    filtered_H1250hill2 = H1250hill[mask2]
    filtered_olddS2Shill2 = olddS2ShillT[mask2]
    
    # Calculate m1
    m1 = np.sum(filtered_H1250hill1 * filtered_olddS2Shill1) / np.sum(filtered_H1250hill1**2)
    m2 = np.sum(filtered_H1250hill2 * filtered_olddS2Shill2) / np.sum(filtered_H1250hill2**2)
    
 
    # Calculate c_low and c_high based on m1
    c_low  = m1* -thresh1
    c_high = m2 * thresh2
    
    H1250hillval = np.asarray(H1250hill)
    olddS2ShillTval = np.asarray(olddS2ShillT)
    
    # Use the new c_low and c_high in TiwariM5
    ans = TiwariM10(H1250model,c_low, m1, m2,c_high,thresh1,thresh2)
    ans1 = TiwariM10(H1250hillval,c_low, m1, m2, c_high,thresh1,thresh2)
    
    ans2 = m1*filtered_H1250hill1
    ans3 = m2*filtered_H1250hill2
    
    ss_residual1 = np.sum((filtered_olddS2Shill1 - ans2)**2)
    ss_residual2 = np.sum((filtered_olddS2Shill2 - ans3)**2)
    n1 = len(filtered_H1250hill1)
    n2 = len(filtered_H1250hill2)
    t_stat1 = m1 / (np.sqrt(ss_residual1 / (n1 - 1)) / np.sqrt(np.sum(filtered_H1250hill1**2)))
    t_stat2 = m2 / (np.sqrt(ss_residual2 / (n2 - 1)) / np.sqrt(np.sum(filtered_H1250hill2**2)))
    p_value1 = 2 * (1 - t.cdf(np.abs(t_stat1), df=n1-1))
    p_value2 = 2 * (1 - t.cdf(np.abs(t_stat2), df=n2-1))
    p_values_dict[im] = {'p_value1': p_value1, 'p_value2': p_value2}
#%% Plotting p-values
pSA_values = list(p_values_dict.keys())
p_value1 = [values['p_value1'] for values in p_values_dict.values()]
p_value2 = [values['p_value2'] for values in p_values_dict.values()]
x_labels = [pSA.replace('pSA_', '') for pSA in pSA_values]
fig,ax   = plt.subplots(figsize=(7.64, 5.67),constrained_layout=True)
ax.text(0.004, 1.05, '(c)', fontsize=22, fontweight='bold', ha='left', va='top')
ax.semilogx(T_RS, p_value1, color = 'g',linewidth=2,label='Lower slope, $h_1$')
ax.semilogx(T_RS, p_value2, color = 'b',linewidth=2,label='Upper slope, $h_2$')
ax.axhline(0.05,color='r',linestyle='--')
ax.set_xlim([0.01,10])
ax.tick_params(direction='in', axis='both', which='both',labelsize=20)
ax.set_xlabel('Vibration Period (s)',size=20)
ax.set_ylabel('p-values',size=20)
ax.legend(fontsize=20)
plt.savefig(os.path.join(out_dir,"p-valueslopes.pdf"))
#%% Make the model
summary_df = pd.read_excel(os.path.join(out_dir,"Bootstrapping_results.xlsx"),index_col=0)
fig,ax = plt.subplots(2,1,figsize=(9.95, 9.04),constrained_layout=True,sharex=True)
ax[0].semilogx(T_RS.values,summary_df['m1_mean'].values,'r',linewidth=2,marker='o',label='Lower slope')
ax[0].semilogx(T_RS.values,summary_df['m2_mean'].values,'b',linewidth=2,marker='s',label='Higher slope')
ax[1].semilogx(T_RS.values,summary_df['clow_mean'].values,'r',linewidth=2,marker='o')
ax[1].semilogx(T_RS.values,summary_df['chigh_mean'].values,'b',linewidth=2,marker='s')
ax[0].legend(fontsize=14,loc='best')
ax[0].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[1].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[1].set_ylabel('Proposed offsets',size=16)
ax[1].set_xlabel('Vibration Period (s)',size=16)
ax[1].set_xlim([0.01,10])
ax[0].set_ylabel('Proposed slope',size=16)
# plt.savefig(os.path.join(out_dir,"SlopeModelInitial.png"))
#%% Smoothing the model & fixing p-values
from scipy.interpolate import interp1d
window_size=10
smooth_summary_df = summary_df.copy()
smooth_summary_df = smooth_summary_df.drop(['m1_std','m2_std','clow_std','chigh_std'],axis=1)
smooth_summary_df['p_value1'] = p_value1
smooth_summary_df['p_value2'] = p_value2
smooth_summary_df['T'] = T_RS
smooth_summary_df['m1_smoothed'] = smooth_summary_df['m1_mean'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')  
smooth_summary_df['m2_smoothed'] = smooth_summary_df['m2_mean'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill') 
transition_period1 = smooth_summary_df[smooth_summary_df['p_value1'] < 0.05]['T'].iloc[0]
transition_period2 = smooth_summary_df[smooth_summary_df['p_value2'] < 0.05]['T'].iloc[0]
start_value1 = 0.1
start_value2 = 0.3
end_value1 = 1
end_value2 = 1.5
smooth_summary_df.loc[smooth_summary_df['T'] < transition_period1, 'm1_smoothed'] = 0
smooth_summary_df.loc[smooth_summary_df['T'] < transition_period2, 'm2_smoothed'] = 0
smooth_summary_df.loc[smooth_summary_df['T'] > end_value2, 'm2_smoothed'] = 0
transition_T_values1 = smooth_summary_df[(smooth_summary_df['T'] >= start_value1) & (smooth_summary_df['T'] <= transition_period1)]['T']
transition_T_values2 = smooth_summary_df[(smooth_summary_df['T'] >= start_value2) & (smooth_summary_df['T'] <= transition_period2)]['T']
transition_T_values3 = smooth_summary_df[(smooth_summary_df['T'] >= end_value1) & (smooth_summary_df['T'] <= end_value2)]['T']
start_slope_value = 0  # starting slope value for the transition
end_slope_value1  = smooth_summary_df['m1_smoothed'][smooth_summary_df['T'] > transition_period1].iloc[0]
end_slope_value2  = smooth_summary_df['m2_smoothed'][smooth_summary_df['T'] > transition_period2].iloc[0]
end_slope_value3  = smooth_summary_df['m2_smoothed'][smooth_summary_df['T'] > end_value1].iloc[0]
interpolator1 = interp1d(
    [start_value1, transition_period1],
    [start_slope_value, end_slope_value1],
    fill_value="extrapolate"
)
interpolator2 = interp1d(
    [start_value2, transition_period2],
    [start_slope_value, end_slope_value2],
    fill_value="extrapolate"
)
interpolator3 = interp1d(
    [end_value1, end_value2],
    [end_slope_value3,start_slope_value],
    fill_value="extrapolate"
)
smooth_summary_df.loc[(smooth_summary_df['T'] >= start_value1) & (smooth_summary_df['T'] <= transition_period1), 'm1_smoothed'] = interpolator1(transition_T_values1)
smooth_summary_df.loc[(smooth_summary_df['T'] >= start_value2) & (smooth_summary_df['T'] <= transition_period2), 'm2_smoothed'] = interpolator2(transition_T_values2)
smooth_summary_df.loc[(smooth_summary_df['T'] >= end_value1) & (smooth_summary_df['T'] <= end_value2), 'm2_smoothed'] = interpolator3(transition_T_values3)
smooth_summary_df['clow_smoothed']  = -thresh1*smooth_summary_df['m1_smoothed']
smooth_summary_df['chigh_smoothed'] = thresh2*smooth_summary_df['m2_smoothed']
fig,ax = plt.subplots(2,1,figsize=(9.95, 9.04),constrained_layout=True,sharex=True)
ax[0].semilogx(T_RS.values,smooth_summary_df['m1_mean'].values,'r',linewidth=1,label='Lower slope')
ax[0].semilogx(T_RS.values,smooth_summary_df['m1_smoothed'].values,'r',linewidth=1,linestyle='--',label='Lower slope smoothed')
ax[0].semilogx(T_RS.values,smooth_summary_df['m2_mean'].values,'b',linewidth=1,label='Higher slope')
ax[0].semilogx(T_RS.values,smooth_summary_df['m2_smoothed'].values,'b',linewidth=1,linestyle='--',label='Higher slope smoothed')
ax[1].semilogx(T_RS.values,smooth_summary_df['clow_mean'].values,'r',linewidth=1)
ax[1].semilogx(T_RS.values,smooth_summary_df['clow_smoothed'].values,'r',linewidth=1,linestyle='--')
ax[1].semilogx(T_RS.values,smooth_summary_df['chigh_mean'].values,'b',linewidth=1)
ax[1].semilogx(T_RS.values,smooth_summary_df['chigh_smoothed'].values,'b',linewidth=1,linestyle='--')
ax[0].legend(fontsize=14,loc='best')
ax[0].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[1].tick_params(direction='in', axis='both', which='both',labelsize=16)
ax[1].set_ylabel('Proposed offsets',size=16)
ax[1].set_xlabel('Vibration Period (s)',size=16)
ax[1].set_xlim([0.01,10])
ax[0].set_ylabel('Proposed slope',size=16)
# plt.savefig(os.path.join(out_dir,"SlopeModelSmoothed.png"))
#%% Model for Paper 1
specific_T_values = [0.01,0.05, 0.1, 0.15,0.2, 0.3, 0.4,  0.5, 0.75, 0.869749002617783,1,1.5,2, 3,4,5,7.5,10] 
filtered_indices = [np.argmin(np.abs(T_RS.values - T)) for T in specific_T_values]
filtered_T_RS = T_RS.values[filtered_indices]
filtered_m1_smoothed = smooth_summary_df['m1_smoothed'].values[filtered_indices]
filtered_m2_smoothed = smooth_summary_df['m2_smoothed'].values[filtered_indices]
fig, ax = plt.subplots(figsize=(6.4,4.8),constrained_layout=True, sharex=True)
# fig.text(0.0, 1, '(b)', fontsize=16, fontweight='bold', ha='left', va='top')
ax.semilogx(filtered_T_RS, filtered_m1_smoothed, 'g', linewidth=2, linestyle='-', label='Lower slope, $m_1$')
ax.semilogx(filtered_T_RS, filtered_m2_smoothed, 'b', linewidth=2, linestyle='-', label='Upper slope, $m_2$')
ax.legend(fontsize=16, loc='best')
ax.tick_params(direction='in', axis='both', which='both', labelsize=20)
ax.set_xlabel('Vibration Period (s)', size=20)
ax.set_xlim([0.01, 10])
ax.set_ylabel('Slope parameters', size=20)
plt.savefig(os.path.join(out_dir, "SlopeModelSmoothed.pdf"))
#%% Ftopo Vs. T for different H1250
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(9.44, 7.39), constrained_layout=True)

SimIMs = pd.read_csv(data_dir1 / "im_sim.csv")
H1250_test = [-50,-30,0,+30,+50]
colors = ['blue', 'red', 'k','blue', 'red']
linestyles = ['-', '-',':', '--','--']
periods = [0.01,0.05, 0.1, 0.15,0.2, 0.3, 0.4,  0.5, 0.75, 0.869749002617783,1,1.5,2, 3,4,5,7.5,10] 
specific_T_values = ['pSA_0.010000000000', 'pSA_0.05', 'pSA_0.1', 'pSA_0.151991108295', 'pSA_0.2', 'pSA_0.3','pSA_0.4', 'pSA_0.5', 'pSA_0.75', 'pSA_0.869749002618', 'pSA_1.0','pSA_1.519911082953', 'pSA_2.0', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 'pSA_7.5', 'pSA_10.0']
adjfactor_dict = {}
for i,H1250 in enumerate(H1250_test):
    adjfactors = []
    for col in specific_T_values:
        # col = f"pSA_{T:.12f}"
        m1 = smooth_summary_df.loc[col]['m1_smoothed']
        m2 = smooth_summary_df.loc[col]['m2_smoothed']
        c_low = smooth_summary_df.loc[col]['clow_smoothed']
        c_high = smooth_summary_df.loc[col]['chigh_smoothed']

        adjfactors.append(
            TiwariM10(float(H1250), c_low, m1, m2, c_high, thresh1, thresh2)
        )
    
    ax.semilogx(periods, adjfactors,  color=colors[i], linestyle=linestyles[i],linewidth=2,label="$H_{1250}$" f" = {H1250}")

ax.legend(fontsize=20, loc='best')
ax.tick_params(direction='in', axis='both', which='both', labelsize=26)
ax.set_xlabel('Vibration Period (s)', size=26)
ax.set_xlim([0.01, 10])
ax.set_ylim([-0.5, 0.5])
ax.tick_params(axis='x', pad=10)
ax.set_ylabel('Adjustment factor, $f_{Hill}$', size=26)
plt.savefig(os.path.join(out_dir, "AdjustmentfactorHill.pdf"))
#%%
fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
specific_T_values_f = [0.05, 0.1]  # floats for computation
specific_T_values = [f"pSA_{t}" for t in specific_T_values_f]  # strings for DataFrame index
filtered_indices = [np.argmin(np.abs(T_RS.values - T)) for T in specific_T_values_f]
filtered_T_RS = T_RS.values[filtered_indices]
for H1250 in H1250_test:
    adjfactors = []
    for col in specific_T_values:   # strings, used to index smooth_summary_df
        m1 = float(smooth_summary_df.loc[col, 'm1_smoothed'])
        m2 = float(smooth_summary_df.loc[col, 'm2_smoothed'])
        c_low = float(smooth_summary_df.loc[col, 'clow_smoothed'])
        c_high = float(smooth_summary_df.loc[col, 'chigh_smoothed'])

        adjfactors.append(
            TiwariM10(float(H1250), c_low, m1, m2, c_high, thresh1, thresh2)
        )
    
    ax.semilogx(filtered_T_RS, adjfactors, label=f"$H_{{1250}}$ = {H1250}")

#%% Applying the factors to original Sims
SimIMs      = pd.read_csv(data_dir1 / "im_sim.csv")
for index,row in SimIMs.iterrows():
    for col in SimIMs.columns[9:]:
        m1      = smooth_summary_df.loc[col]['m1_smoothed']
        m2      = smooth_summary_df.loc[col]['m2_smoothed']
        c_low      = smooth_summary_df.loc[col]['clow_smoothed']
        c_high     = smooth_summary_df.loc[col]['chigh_smoothed']
        adjfactor = []
        station=row['stat_id']
        ind = stations.index[stations['stat_id']==station][0]
        H1250      = stations.iloc[ind]['H1250']
        stat_group = stations.iloc[ind]['Geomorphology']
        if stat_group == "Hill":
            adjfactor = TiwariM10(H1250,c_low,m1,m2,c_high,thresh1,thresh2)
        else:
            adjfactor = 0
        SimIMs.at[index,col]*=np.exp(adjfactor)    
Simadj_fp = os.path.join("Model Inputs", "im_simM10SlopeModel3.csv")
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
sim_ffp = data_dir / "im_simM10SlopeModel3.csv"
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
results.event_res_df.to_csv(output_dir / "event_residualsM10SlopeModel3.csv", index_label="event_id")
results.rem_res_df.to_csv(output_dir / "remaining_residualsM10SlopeModel3.csv", index_label="gm_id")
results.bias_std_df.to_csv(output_dir / "bias_stdM10SlopeModel3.csv", index_label="IM")
results.site_res_df.to_csv(output_dir / "site_residualsM10SlopeModel3.csv", index_label="stat_id")
results.site_cond_std_df.to_csv(output_dir / "station_cond_stdM10SlopeModel3.csv", index_label="stat_id")
#%% Reduction in phiS2S - All sites
output_dir           = Path(os.path.join(os.getcwd(),"Model Outputs"))
varcomps_all_new     = pd.read_csv(Path(output_dir/"bias_stdM10SlopeModel3.csv"),index_col=0)
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
#%% Computation of a+dS2S for hill sites
aold            = varcomps.iloc[6:,varcomps.columns.get_loc('bias')]
anew            = varcomps_all_new.iloc[6:,varcomps_all_new.columns.get_loc('bias')]
newdS2Sres_all  = pd.read_csv(output_dir / "site_residualsM10SlopeModel3.csv",index_col=0)
newdS2Sres_all  = newdS2Sres_all.dropna(axis=1)
newdS2Sres_all  = newdS2Sres_all.iloc[:,6:]
sorted_columns  = sorted(newdS2Sres_all.columns, key=lambda x: float(x.split('_')[1]))
newdS2Sres_all  = newdS2Sres_all[sorted_columns]
anew            = anew[sorted_columns]
newdS2Shill     = newdS2Sres_all.loc[nstationsmod[(nstationsmod['Geomorphology'] == "Hill")]['stat_id']]
oldadS2Shill    = aold+olddS2Shill
newadS2Shill    = anew+newdS2Shill
fig,ax          = plt.subplots(figsize=(8.52,6.19),constrained_layout=True)
for i in range(len(newadS2Shill)):
    ax.semilogx(T_RS.values,newadS2Shill.iloc[i,:],'gray',linewidth=2)
ax.semilogx(T_RS.values,oldadS2Shill.mean(),'r',label='Original mean')
ax.semilogx(T_RS.values,newadS2Shill.mean(),'b',label='After adjustments mean')
ax.semilogx(T_RS.values,newadS2Shill.mean()+newadS2Shill.std(),'b--')
ax.semilogx(T_RS.values,newadS2Shill.mean()-newadS2Shill.std(),'b--')
ax.axhline(0,color='maroon',linestyle='--')
ax.grid(True, color='gray',which='both', linestyle='--', linewidth=0.5)
ax.legend(fontsize=16)
ax.set_ylabel('Systematic residual $a\ +\ \delta S2S_{S}$ for hill sites',size=16)
ax.set_xlabel('Vibration Period, T (s)',size=16)
ax.tick_params(labelsize=14,direction='in',axis='both',which='both')
ax.set_xlim([0.01,10])
plt.savefig(os.path.join(out_dir,'Sysresidualhillsites.png'))
#%% Cat. Mean and cat standard deviation plots (Just dS2S plots)
newdS2Serr_all  = pd.read_csv(output_dir / "station_cond_stdM10SlopeModel3.csv",index_col=0)
newdS2Serr_all  = newdS2Serr_all.dropna(axis=1)
newdS2Serr_all  = newdS2Serr_all.iloc[:,6:]
sorted_columns  = sorted(newdS2Serr_all.columns, key=lambda x: float(x.split('_')[1]))
newdS2Serr_all  = newdS2Serr_all[sorted_columns]
newdS2Serrhill  = newdS2Serr_all.loc[nstationsmod[(nstationsmod['Geomorphology'] == "Hill")]['stat_id']]
oldstdall       = np.sqrt(np.mean(dS2Sres_all**2,axis=0)+np.mean(dS2Serr_all**2,axis=0))
newstdall       = np.sqrt(np.mean(newdS2Sres_all**2,axis=0)+np.mean(newdS2Serr_all**2,axis=0))
oldhillstd      = np.sqrt(np.mean(olddS2Shill**2,axis=0)+np.mean(olddS2Serrhill**2,axis=0))
newhillstd      = np.sqrt(np.mean(newdS2Shill**2,axis=0)+np.mean(newdS2Serrhill**2,axis=0))
fig,axes    = plt.subplots(2,1,figsize = (8.52, 7.62),constrained_layout=True,sharex=True)
axes[0].semilogx(T_RS.values,olddS2Shill.mean(),'g',linewidth=4,label='Hill sites (Original)')
axes[0].semilogx(T_RS.values,newdS2Shill.mean(),'g--',linewidth=4,label='Hill sites (Proposed model)')
axes[0].axhline(0,color='k',linestyle='--',linewidth=4)
axes[0].axhline(0,color='k',linestyle='-',linewidth=2)
axes[1].semilogx(T_RS.values,oldstdall,'k',linewidth=4,label='All sites (Original)')
axes[1].semilogx(T_RS.values,oldhillstd,'g',linewidth=4,label='Hill sites (Original)')
axes[1].semilogx(T_RS.values,newstdall,'k--',linewidth=4,label='All sites (Proposed model)')
axes[1].semilogx(T_RS.values,newhillstd,'g--',linewidth=4,label='Hill sites (Proposed model)')
axes[0].text(0.0105,0.02,'Underprediction',size=16,fontweight='bold')
axes[0].text(0.0105,-0.1,'Overprediction',size=16,fontweight='bold')
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
plt.savefig(os.path.join(out_dir,'deltaS2Shillsites.png'))
#%% Cat. Mean and cat standard deviation plots
plt.rcParams["font.family"] = "Times New Roman"
Allsitesres     = pd.read_csv(Path("Results_combined.csv"),index_col=0)
newdS2Serr_all  = pd.read_csv(output_dir / "station_cond_stdM10SlopeModel3.csv",index_col=0)
newdS2Serr_all  = newdS2Serr_all.dropna(axis=1)
newdS2Serr_all  = newdS2Serr_all.iloc[:,6:]
sorted_columns  = sorted(newdS2Serr_all.columns, key=lambda x: float(x.split('_')[1]))
newdS2Serr_all  = newdS2Serr_all[sorted_columns]
newdS2Serrhill  = newdS2Serr_all.loc[nstationsmod[(nstationsmod['Geomorphology'] == "Hill")]['stat_id']]
oldstdall       = np.sqrt(np.mean(dS2Sres_all**2,axis=0)+np.mean(dS2Serr_all**2,axis=0))
newstdall       = np.sqrt(np.mean(newdS2Sres_all**2,axis=0)+np.mean(newdS2Serr_all**2,axis=0))
oldhillstd      = np.sqrt(np.mean(olddS2Shill**2,axis=0)+np.mean(olddS2Serrhill**2,axis=0))
newhillstd      = np.sqrt(np.mean(newdS2Shill**2,axis=0)+np.mean(newdS2Serrhill**2,axis=0))
fig,axes    = plt.subplots(2,1,figsize = (11.21,  9.54),constrained_layout=True,sharex=True)
fig.text(0.03, 1, '(a)', fontsize=36, fontweight='bold', ha='left', va='top')
axes[0].semilogx(T_RS.values,oldadS2Shill.mean(),'#993333',linewidth=2,label='Hill sites (Original)')
axes[0].semilogx(T_RS.values,newadS2Shill.mean(),'#993333',linestyle='--',linewidth=2,label='Hill sites (Proposed model)')
axes[0].axhline(0,color='k',linestyle='--',linewidth=1)
axes[0].semilogx(T_RS.values,aold+dS2Sres_all.mean(),'k',linewidth=2)
axes[0].semilogx(T_RS.values,Allsitesres['anew'].values+newdS2Sres_all.mean(),'k--',linewidth=2)
axes[1].semilogx(T_RS.values,oldhillstd,'#993333',linewidth=2,label='Hill sites (Original)')
axes[1].semilogx(T_RS.values,oldstdall,'k',linewidth=2,label='All sites (Original)')
axes[1].semilogx(T_RS.values,newhillstd,'#993333',linestyle='--',linewidth=2,label='Hill sites (Proposed model)')
axes[1].semilogx(T_RS.values,Allsitesres['newstd'].values,'k--',linewidth=2,label='All sites (All parametric models)')

axes[0].text(0.0105,0.12,'Underprediction',size=30,fontstyle='italic')
axes[0].text(0.0105,-0.27,'Overprediction',size=30,fontstyle='italic')
axes[0].set_ylabel('$a\ +\ \delta S2S_{S}$',size=38)
axes[0].tick_params(labelsize=32,direction='in', axis='both', which='both')
axes[0].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)   
axes[1].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
axes[1].set_ylabel('$\phi_{S2S}$',size=38)
axes[1].set_xlabel('Vibration Period, T (s)', size=38)
axes[1].legend(fontsize=26,frameon=False)
axes[1].tick_params(labelsize=32,direction='in', axis='both', which='both')
axes[1].set_xlim([0.01, 10])
axes[1].tick_params(axis='x', pad=10)
axes[1].set_ylim([0,None])
plt.savefig(os.path.join(out_dir, 'Sysresidual&stdhillsites.pdf'))
#%% % Reduction in phiS2S
fig,ax = plt.subplots(figsize=(7.43, 6.0),constrained_layout=True)
fig.text(0.0, 1, '(b)', fontsize=22, fontweight='bold', ha='left', va='top')
Red = ((oldhillstd-newhillstd)/oldhillstd)*100
ax.semilogx(T_RS.values,Red,'#993333',linewidth=2,label='Hill sites')
ax.semilogx(T_RS.values,Allsitesres['Reduction'],'k',linewidth=2, label = 'All sites')
ax.set_xlabel('Vibration Period, T (s)', size=22)
ax.set_ylabel('Reduction in $\phi_{S2S}$',size=22)
ax.tick_params(labelsize=20,direction='in', axis='both', which='both')
ax.set_xlim([0.01, 10])
ax.legend(fontsize=20,frameon=False)
ax.grid(True, color='gray',which='both', linestyle='--', linewidth=0.4,alpha=0.5)
ax.set_ylim([0,None])
ax.tick_params(axis='x', pad=5)
plt.savefig(os.path.join(out_dir,'RedinphiS2Shill.pdf'))
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
#%% Lowess plots before and after
# IMs        = ['pSA_0.05','pSA_0.1','pSA_0.2','pSA_0.4'] 
IMs       = ['pSA_0.3','pSA_0.75','pSA_1.0','pSA_3.0']
# IMs        = ['pSA_1.519911082953','pSA_2.0','pSA_3.0','pSA_5.0']
# IMs          = ['pSA_6.135907273413','pSA_7.5','pSA_8.697490026178','pSA_10.0']
fig,ax = plt.subplots(2,2,figsize=(12.06,8.05),constrained_layout=True,sharex=True,sharey=True)
H1250hill = np.array(H1250hill)
order = np.argsort(H1250hill)
for idx, im in enumerate(IMs):
    i, j = divmod(idx, 2) 
    lowessold,lowessoldstd = loess(H1250hill,olddS2Shill.loc[:,im].values,0.3)
    lowessnew,lowessnewstd = loess(H1250hill,newdS2Shill.loc[:,im].values,0.3)
    ax[i, j].scatter(H1250hill, newdS2Shill.loc[:, im], color='#008080', s=40,edgecolor='k')
    ax[i,j].plot(H1250hill[order], lowessold[order], color='#FF6F61', linewidth=2, label='Before adjustment')
    ax[i,j].fill_between(H1250hill[order], lowessold[order] - 1.96*lowessoldstd[order], lowessold[order] + 1.96*lowessoldstd[order], color='red', alpha=0.1)
    ax[i,j].plot(H1250hill[order], lowessnew[order], color='#008080',linewidth=2, label='After adjustment')
    ax[i,j].fill_between(H1250hill[order], lowessnew[order] - 1.96*lowessnew[order], lowessnew[order] + 1.96*lowessnew[order], color='blue', alpha=0.1)
    ax[i, j].axhline(0, color='k', linestyle='--')
    rounded_value = np.round(float(im.split('pSA_')[1]), 2)
    ax[i, j].set_ylabel(f'$\delta S2S$ (T={rounded_value}s)', size=20)
    ax[i, j].tick_params(direction='in', axis='both', which='both', labelsize=20)
    ax[i, j].axvline(0, color='k', linestyle='--')
    # ax[i, j].axvline(thresh2, color='k', linestyle='--')
    x_min, x_max = ax[i,j].get_xlim()
    max_x_range = max(abs(x_max - x0), abs(x_min - x0))
    ax[i, j].set_xlim([x0 - max_x_range, x0 + max_x_range])
global_y_min, global_y_max = float('inf'), float('-inf')
for i in range(2):  # rows and cols define the subplot grid
    for j in range(2):
         y_min, y_max = ax[i, j].get_ylim()
         global_y_min = min(global_y_min, y_min)
         global_y_max = max(global_y_max, y_max)

y0= 0
max_y_range = max(abs(global_y_max - y0), abs(global_y_min - y0))
for i in range(2):
    for j in range(2):
        ax[i, j].set_ylim([y0 - max_y_range, y0 + max_y_range])

ax[0][0].legend(fontsize=16)  
ax[1][0].set_xlabel('$H_{1250}$ (m)', size=20)
ax[1][1].set_xlabel('$H_{1250}$ (m)', size=20)

# plt.savefig(os.path.join(out_dir,"Loessbefore_afterlong.png"))
plt.savefig(os.path.join(out_dir,"Loessbefore_aftershort.pdf"))
# plt.savefig(os.path.join(out_dir,"Loessbefore_afterverylong.png"))
# plt.savefig(os.path.join(out_dir,"Loessbefore_afterveryshort.png"))